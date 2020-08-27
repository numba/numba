from numba import cuda, float32, int32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase


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


if __name__ == '__main__':
    unittest.main()
