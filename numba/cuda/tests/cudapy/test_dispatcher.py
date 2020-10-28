import numpy as np
import threading

from numba import cuda, float32, int32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase


def add(x, y):
    return x + y


def add_kernel(r, x, y):
    r[0] = x + y


@skip_on_cudasim('Dispatcher objects not used in the simulator')
class TestDispatcher(CUDATestCase):
    def _test_no_double_specialize(self, dispatcher, ty):

        with self.assertRaises(RuntimeError) as e:
            dispatcher.specialize(ty)

        self.assertIn('Dispatcher already specialized', str(e.exception))

    def test_no_double_specialize_sig_same_types(self):
        # Attempting to specialize a kernel jitted with a signature is illegal,
        # even for the same types the kernel is already specialized for.
        @cuda.jit('void(float32[::1])')
        def f(x):
            pass

        self._test_no_double_specialize(f, float32[::1])

    def test_no_double_specialize_no_sig_same_types(self):
        # Attempting to specialize an already-specialized kernel is illegal,
        # even for the same types the kernel is already specialized for.
        @cuda.jit
        def f(x):
            pass

        f_specialized = f.specialize(float32[::1])
        self._test_no_double_specialize(f_specialized, float32[::1])

    def test_no_double_specialize_sig_diff_types(self):
        # Attempting to specialize a kernel jitted with a signature is illegal.
        @cuda.jit('void(int32[::1])')
        def f(x):
            pass

        self._test_no_double_specialize(f, float32[::1])

    def test_no_double_specialize_no_sig_diff_types(self):
        # Attempting to specialize an already-specialized kernel is illegal.
        @cuda.jit
        def f(x):
            pass

        f_specialized = f.specialize(int32[::1])
        self._test_no_double_specialize(f_specialized, float32[::1])

    def test_specialize_cache_same(self):
        # Ensure that the same dispatcher is returned for the same argument
        # types, and that different dispatchers are returned for different
        # argument types.
        @cuda.jit
        def f(x):
            pass

        self.assertEqual(len(f.specializations), 0)

        f_float32 = f.specialize(float32[::1])
        self.assertEqual(len(f.specializations), 1)

        f_float32_2 = f.specialize(float32[::1])
        self.assertEqual(len(f.specializations), 1)
        self.assertIs(f_float32, f_float32_2)

        f_int32 = f.specialize(int32[::1])
        self.assertEqual(len(f.specializations), 2)
        self.assertIsNot(f_int32, f_float32)

    def test_specialize_cache_same_with_ordering(self):
        # Ensure that the same dispatcher is returned for the same argument
        # types, and that different dispatchers are returned for different
        # argument types, taking into account array ordering and multiple
        # arguments.
        @cuda.jit
        def f(x, y):
            pass

        self.assertEqual(len(f.specializations), 0)

        # 'A' order specialization
        f_f32a_f32a = f.specialize(float32[:], float32[:])
        self.assertEqual(len(f.specializations), 1)

        # 'C' order specialization
        f_f32c_f32c = f.specialize(float32[::1], float32[::1])
        self.assertEqual(len(f.specializations), 2)
        self.assertIsNot(f_f32a_f32a, f_f32c_f32c)

        # Reuse 'C' order specialization
        f_f32c_f32c_2 = f.specialize(float32[::1], float32[::1])
        self.assertEqual(len(f.specializations), 2)
        self.assertIs(f_f32c_f32c, f_f32c_f32c_2)

    # The following tests are based on those in numba.tests.test_dispatcher

    def test_coerce_input_types(self):
        # Do not allow unsafe conversions if we can still compile other
        # specializations.
        c_add = cuda.jit(add_kernel)

        # Using a complex128 allows us to represent any result produced by the
        # test
        r = np.zeros(1, dtype=np.complex128)

        c_add[1, 1](r, 123, 456)
        self.assertEqual(r[0], add(123, 456))

        c_add[1, 1](r, 12.3, 45.6)
        self.assertEqual(r[0], add(12.3, 45.6))

        c_add[1, 1](r, 12.3, 45.6j)
        self.assertEqual(r[0], add(12.3, 45.6j))

        c_add[1, 1](r, 12300000000, 456)
        self.assertEqual(r[0], add(12300000000, 456))

        # Now force compilation of only a single specialization
        c_add = cuda.jit('(i4[::1], i4, i4)')(add_kernel)
        r = np.zeros(1, dtype=np.int32)

        c_add[1, 1](r, 123, 456)
        self.assertPreciseEqual(r[0], add(123, 456))

    @unittest.expectedFailure
    def test_coerce_input_types_unsafe(self):
        # Implicit (unsafe) conversion of float to int, originally from
        # test_coerce_input_types. This test presently fails with the CUDA
        # Dispatcher because argument preparation is done by
        # _Kernel._prepare_args, which is currently inflexible with respect to
        # the types it can accept when preparing.
        #
        # This test is marked as xfail until future changes enable this
        # behavior.
        c_add = cuda.jit('(i4[::1], i4, i4)')(add_kernel)
        r = np.zeros(1, dtype=np.int32)

        c_add[1, 1](r, 12.3, 45.6)
        self.assertPreciseEqual(r[0], add(12, 45))

    def test_coerce_input_types_unsafe_complex(self):
        # Implicit conversion of complex to int disallowed
        c_add = cuda.jit('(i4[::1], i4, i4)')(add_kernel)
        r = np.zeros(1, dtype=np.int32)

        with self.assertRaises(TypeError):
            c_add[1, 1](r, 12.3, 45.6j)

    def test_ambiguous_new_version(self):
        """Test compiling new version in an ambiguous case
        """
        @cuda.jit
        def foo(r, a, b):
            r[0] = a + b

        r = np.zeros(1, dtype=np.float64)
        INT = 1
        FLT = 1.5

        foo[1, 1](r, INT, FLT)
        self.assertAlmostEqual(r[0], INT + FLT)
        self.assertEqual(len(foo.overloads), 1)

        foo[1, 1](r, FLT, INT)
        self.assertAlmostEqual(r[0], FLT + INT)
        self.assertEqual(len(foo.overloads), 2)

        foo[1, 1](r, FLT, FLT)
        self.assertAlmostEqual(r[0], FLT + FLT)
        self.assertEqual(len(foo.overloads), 3)

        # The following call is ambiguous because (int, int) can resolve
        # to (float, int) or (int, float) with equal weight.
        foo[1, 1](r, 1, 1)
        self.assertAlmostEqual(r[0], INT + INT)
        self.assertEqual(len(foo.overloads), 4, "didn't compile a new "
                                                "version")

    def test_lock(self):
        """
        Test that (lazy) compiling from several threads at once doesn't
        produce errors (see issue #908).
        """
        errors = []

        @cuda.jit
        def foo(r, x):
            r[0] = x + 1

        def wrapper():
            try:
                r = np.zeros(1, dtype=np.int64)
                foo[1, 1](r, 1)
                self.assertEqual(r[0], 2)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=wrapper) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors)


if __name__ == '__main__':
    unittest.main()
