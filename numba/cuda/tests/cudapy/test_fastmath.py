from numba import cuda, float32
from math import cos, sin, tan, exp, log, log10, log2, pow
import numpy as np
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import unittest


@skip_on_cudasim('Fastmath and PTX inspection not available on cudasim')
class TestFastMathOption(CUDATestCase):
    def test_kernel(self):
        # Test the cast of an int being used in fastmath divide
        def foo(arr, val):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] = float32(i) / val

        sig = (float32[::1], float32)
        fastver = cuda.jit(sig, fastmath=True)(foo)
        precver = cuda.jit(sig)(foo)

        self.assertIn('div.approx.ftz.f32', fastver.ptx[sig])
        self.assertNotIn('div.approx.ftz.f32', precver.ptx[sig])

    def test_cosf(self):
        def f1(r, x):
            r[0] = cos(x)

        sig = (float32[::1], float32)
        fastver = cuda.jit(sig, fastmath=True)(f1)
        slowver = cuda.jit(sig)(f1)
        self.assertIn('cos.approx.ftz.f32 ', fastver.ptx[sig])
        self.assertNotIn('cos.approx.ftz.f32 ', slowver.ptx[sig])

    def test_sinf(self):
        def f2(r, x):
            r[0] = sin(x)

        sig = (float32[::1], float32)
        fastver = cuda.jit(sig, fastmath=True)(f2)
        slowver = cuda.jit(sig)(f2)
        self.assertIn('sin.approx.ftz.f32 ', fastver.ptx[sig])
        self.assertNotIn('sin.approx.ftz.f32 ', slowver.ptx[sig])

    def test_tanf(self):
        def f3(r, x):
            r[0] = tan(x)

        sig = (float32[::1], float32)
        fastver = cuda.jit(sig, fastmath=True)(f3)
        slowver = cuda.jit(sig)(f3)
        self.assertIn('sin.approx.ftz.f32 ', fastver.ptx[sig])
        self.assertIn('cos.approx.ftz.f32 ', fastver.ptx[sig])
        self.assertIn('div.approx.ftz.f32 ', fastver.ptx[sig])
        self.assertNotIn('sin.approx.ftz.f32 ', slowver.ptx[sig])

    def test_expf(self):
        def f4(r, x):
            r[0] = exp(x)

        sig = (float32[::1], float32)
        fastver = cuda.jit(sig, fastmath=True)(f4)
        slowver = cuda.jit(sig)(f4)
        self.assertNotIn('fma.rn.f32 ', fastver.ptx[sig])
        self.assertIn('fma.rn.f32 ', slowver.ptx[sig])

    def test_logf(self):
        def f5(r, x):
            r[0] = log(x)

        sig = (float32[::1], float32)
        fastver = cuda.jit(sig, fastmath=True)(f5)
        slowver = cuda.jit(sig)(f5)
        self.assertIn('lg2.approx.ftz.f32 ', fastver.ptx[sig])
        # Look for constant used to convert from log base 2 to log base e
        self.assertIn('0f3F317218', fastver.ptx[sig])
        self.assertNotIn('lg2.approx.ftz.f32 ', slowver.ptx[sig])

    def test_log10f(self):
        def f6(r, x):
            r[0] = log10(x)

        sig = (float32[::1], float32)
        fastver = cuda.jit(sig, fastmath=True)(f6)
        slowver = cuda.jit(sig)(f6)
        self.assertIn('lg2.approx.ftz.f32 ', fastver.ptx[sig])
        # Look for constant used to convert from log base 2 to log base 10
        self.assertIn('0f3E9A209B', fastver.ptx[sig])
        self.assertNotIn('lg2.approx.ftz.f32 ', slowver.ptx[sig])

    def test_log2f(self):
        def f7(r, x):
            r[0] = log2(x)

        sig = (float32[::1], float32)
        fastver = cuda.jit(sig, fastmath=True)(f7)
        slowver = cuda.jit(sig)(f7)
        self.assertIn('lg2.approx.ftz.f32 ', fastver.ptx[sig])
        self.assertNotIn('lg2.approx.ftz.f32 ', slowver.ptx[sig])

    def test_powf(self):
        def f8(r, x, y):
            r[0] = pow(x, y)

        sig = (float32[::1], float32, float32)
        fastver = cuda.jit(sig, fastmath=True)(f8)
        slowver = cuda.jit(sig)(f8)
        self.assertIn('lg2.approx.ftz.f32 ', fastver.ptx[sig])
        self.assertNotIn('lg2.approx.ftz.f32 ', slowver.ptx[sig])

    def test_divf(self):
        def f9(r, x, y):
            r[0] = x / y

        sig = (float32[::1], float32, float32)
        fastver = cuda.jit(sig, fastmath=True)(f9)
        slowver = cuda.jit(sig)(f9)
        self.assertIn('div.approx.ftz.f32 ', fastver.ptx[sig])
        self.assertNotIn('div.approx.ftz.f32 ', slowver.ptx[sig])
        self.assertIn('div.rn.f32', slowver.ptx[sig])
        self.assertNotIn('div.rn.f32', fastver.ptx[sig])

    def test_divf_exception(self):
        def f10(r, x, y):
            r[0] = x / y

        sig = (float32[::1], float32, float32)
        fastver = cuda.jit(sig, fastmath=True, debug=True)(f10)
        slowver = cuda.jit(sig, debug=True)(f10)
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

        sig = (float32[::1], float32)
        fastver = cuda.jit(sig, fastmath=True)(bar)
        precver = cuda.jit(sig)(bar)

        self.assertIn('div.full.ftz.f32', fastver.ptx[sig])
        self.assertNotIn('div.full.ftz.f32', precver.ptx[sig])


if __name__ == '__main__':
    unittest.main()
