from numba import cuda, float32
from math import cos, sin, tan, exp, log, log10, log2, pow
import numpy as np
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import unittest


class TestFastMathOption(CUDATestCase):
    @skip_on_cudasim('fast divide not available in CUDASIM')
    def test_kernel(self):
        # Test the cast of an int being used in fastmath divide
        def foo(arr, val):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] = float32(i) / val

        fastver = cuda.jit("void(float32[:], float32)", fastmath=True)(foo)
        precver = cuda.jit("void(float32[:], float32)")(foo)

        self.assertIn('div.approx.ftz.f32', fastver.ptx)
        self.assertNotIn('div.approx.ftz.f32', precver.ptx)

    @skip_on_cudasim('fast cos not available in CUDASIM')
    def test_cosf(self):
        def f1(r, x):
            r[0] = cos(x)

        fastver = cuda.jit("void(float32[::1], float32)", fastmath=True)(f1)
        slowver = cuda.jit("void(float32[::1], float32)")(f1)
        self.assertIn('cos.approx.ftz.f32 ', fastver.ptx)
        self.assertNotIn('cos.approx.ftz.f32 ', slowver.ptx)

    @skip_on_cudasim('fast sin not available in CUDASIM')
    def test_sinf(self):
        def f2(r, x):
            r[0] = sin(x)

        fastver = cuda.jit("void(float32[::1], float32)", fastmath=True)(f2)
        slowver = cuda.jit("void(float32[::1], float32)")(f2)
        self.assertIn('sin.approx.ftz.f32 ', fastver.ptx)
        self.assertNotIn('sin.approx.ftz.f32 ', slowver.ptx)

    @skip_on_cudasim('fast tan not available in CUDASIM')
    def test_tanf(self):
        def f3(r, x):
            r[0] = tan(x)

        fastver = cuda.jit("void(float32[::1], float32)", fastmath=True)(f3)
        slowver = cuda.jit("void(float32[::1], float32)")(f3)
        self.assertIn('sin.approx.ftz.f32 ', fastver.ptx)
        self.assertIn('cos.approx.ftz.f32 ', fastver.ptx)
        self.assertIn('div.approx.ftz.f32 ', fastver.ptx)
        self.assertNotIn('sin.approx.ftz.f32 ', slowver.ptx)

    @skip_on_cudasim('fast exp not available in CUDASIM')
    def test_expf(self):
        def f4(r, x):
            r[0] = exp(x)

        fastver = cuda.jit("void(float32[::1], float32)", fastmath=True)(f4)
        slowver = cuda.jit("void(float32[::1], float32)")(f4)
        self.assertNotIn('fma.rn.f32 ', fastver.ptx)
        self.assertIn('fma.rn.f32 ', slowver.ptx)

    @skip_on_cudasim('fast log not available in CUDASIM')
    def test_logf(self):
        def f5(r, x):
            r[0] = log(x)

        fastver = cuda.jit("void(float32[::1], float32)", fastmath=True)(f5)
        slowver = cuda.jit("void(float32[::1], float32)")(f5)
        self.assertIn('lg2.approx.ftz.f32 ', fastver.ptx)
        # Look for constant used to convert from log base 2 to log base e
        self.assertIn('0f3F317218', fastver.ptx)
        self.assertNotIn('lg2.approx.ftz.f32 ', slowver.ptx)

    @skip_on_cudasim('fast log10 not available in CUDASIM')
    def test_log10f(self):
        def f6(r, x):
            r[0] = log10(x)

        fastver = cuda.jit("void(float32[::1], float32)", fastmath=True)(f6)
        slowver = cuda.jit("void(float32[::1], float32)")(f6)
        self.assertIn('lg2.approx.ftz.f32 ', fastver.ptx)
        # Look for constant used to convert from log base 2 to log base 10
        self.assertIn('0f3E9A209B', fastver.ptx)
        self.assertNotIn('lg2.approx.ftz.f32 ', slowver.ptx)

    @skip_on_cudasim('fast log2 not available in CUDASIM')
    def test_log2f(self):
        def f7(r, x):
            r[0] = log2(x)

        fastver = cuda.jit("void(float32[::1], float32)", fastmath=True)(f7)
        slowver = cuda.jit("void(float32[::1], float32)")(f7)
        self.assertIn('lg2.approx.ftz.f32 ', fastver.ptx)
        self.assertNotIn('lg2.approx.ftz.f32 ', slowver.ptx)

    @skip_on_cudasim('fast pow not available in CUDASIM')
    def test_powf(self):
        def f8(r, x, y):
            r[0] = pow(x, y)

        fastver = cuda.jit("void(float32[::1], float32, float32)",
                           fastmath=True)(f8)
        slowver = cuda.jit("void(float32[::1], float32, float32)")(f8)
        self.assertIn('lg2.approx.ftz.f32 ', fastver.ptx)
        self.assertNotIn('lg2.approx.ftz.f32 ', slowver.ptx)

    @skip_on_cudasim('fast divide not available in CUDASIM')
    def test_divf(self):
        def f9(r, x, y):
            r[0] = x / y

        fastver = cuda.jit("void(float32[::1], float32, float32)",
                           fastmath=True)(f9)
        slowver = cuda.jit("void(float32[::1], float32, float32)")(f9)
        self.assertIn('div.approx.ftz.f32 ', fastver.ptx)
        self.assertNotIn('div.approx.ftz.f32 ', slowver.ptx)
        self.assertIn('div.rn.f32', slowver.ptx)
        self.assertNotIn('div.rn.f32', fastver.ptx)

    @skip_on_cudasim('fast divide not available in CUDASIM')
    def test_divf_exception(self):
        def f10(r, x, y):
            r[0] = x / y

        fastver = cuda.jit("void(float32[::1], float32, float32)",
                           fastmath=True, debug=True)(f10)
        slowver = cuda.jit("void(float32[::1], float32, float32)",
                           debug=True)(f10)
        nelem = 10
        ary = np.empty(nelem, dtype=np.float32)
        with self.assertRaises(ZeroDivisionError):
            slowver[1, nelem](ary, 10.0, 0.0)

        try:
            fastver[1, nelem](ary, 10.0, 0.0)
        except ZeroDivisionError:
            self.fail("Divide in fastmath should not throw ZeroDivisionError")

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
