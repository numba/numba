from numba import cuda, float32
from numba.cuda.testing import CUDATestCase
import unittest


class TestFastMathOption(CUDATestCase):
    def test_kernel(self):

        def foo(arr, val):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] = float32(i) / val

        fastver = cuda.jit("void(float32[:], float32)", fastmath=True)(foo)
        precver = cuda.jit("void(float32[:], float32)")(foo)

        self.assertIn('div.full.ftz.f32', fastver.ptx)
        self.assertNotIn('div.full.ftz.f32', precver.ptx)

    def test_device(self):
        # fastmath option is ignored for device function
        @cuda.jit("float32(float32, float32)", device=True)
        def foo(a, b):
            return a / b

        def bar(arr, val):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] = foo(i, val)

        fastver = cuda.jit("void(float32[:], float32)", fastmath=True)(bar)
        precver = cuda.jit("void(float32[:], float32)")(bar)

        self.assertIn('div.full.ftz.f32', fastver.ptx)
        self.assertNotIn('div.full.ftz.f32', precver.ptx)


if __name__ == '__main__':
    unittest.main()
