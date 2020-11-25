from numba import cuda, float32, float64, int32, int64, void
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import numpy as np
import math


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

    def test_get_regs_per_thread_unspecialized(self):
        # A kernel where the register usage per thread is likely to differ
        # between different specializations
        @cuda.jit
        def pi_sin_array(x, n):
            i = cuda.grid(1)
            if i < n:
                x[i] = 3.14 * math.sin(x[i])

        # Call the kernel with different arguments to create two different
        # definitions within the Dispatcher object
        N = 10
        arr_f32 = np.zeros(N, dtype=np.float32)
        arr_f64 = np.zeros(N, dtype=np.float64)

        pi_sin_array[1, N](arr_f32, N)
        pi_sin_array[1, N](arr_f64, N)

        # Check we get a positive integer for the two different variations
        sig_f32 = void(float32[::1], int64)
        sig_f64 = void(float64[::1], int64)
        regs_per_thread_f32 = pi_sin_array.get_regs_per_thread(sig_f32)
        regs_per_thread_f64 = pi_sin_array.get_regs_per_thread(sig_f64)

        self.assertIsInstance(regs_per_thread_f32, int)
        self.assertIsInstance(regs_per_thread_f64, int)

        self.assertGreater(regs_per_thread_f32, 0)
        self.assertGreater(regs_per_thread_f64, 0)

        # Check that getting the registers per thread for all signatures
        # provides the same values as getting the registers per thread for
        # individual signatures. Note that the returned dict is indexed by
        # (cc, argtypes) pairs (in keeping with definitions, ptx, LLVM IR,
        # etc.)
        regs_per_thread_all = pi_sin_array.get_regs_per_thread()
        cc = cuda.current_context().device.compute_capability
        self.assertEqual(regs_per_thread_all[cc, sig_f32.args],
                         regs_per_thread_f32)
        self.assertEqual(regs_per_thread_all[cc, sig_f64.args],
                         regs_per_thread_f64)

        if regs_per_thread_f32 == regs_per_thread_f64:
            # If the register usage is the same for both variants, there may be
            # a bug, but this may also be an artifact of the compiler / driver
            # / device combination, so produce an informational message only.
            print('f32 and f64 variant thread usages are equal.')
            print('This may warrant some investigation. Devices:')
            cuda.detect()

    def test_get_regs_per_thread_specialized(self):
        @cuda.jit(void(float32[::1], int64))
        def pi_sin_array(x, n):
            i = cuda.grid(1)
            if i < n:
                x[i] = 3.14 * math.sin(x[i])

        # Check we get a positive integer for the specialized variation
        regs_per_thread = pi_sin_array.get_regs_per_thread()
        self.assertIsInstance(regs_per_thread, int)
        self.assertGreater(regs_per_thread, 0)


if __name__ == '__main__':
    unittest.main()
