from dataclasses import dataclass, field
from numba import cuda, float32
from numba.cuda.compiler import compile_ptx_for_current_device
from math import cos, sin, tan, exp, log, log10, log2, pow
import numpy as np
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import unittest

@dataclass
class FastMathCriterion:
    fast_expected: list[str] = field(default_factory=list)
    fast_unexpected: list[str] = field(default_factory=list)
    slow_expected: list[str] = field(default_factory=list)
    slow_unexpected: list[str] = field(default_factory=list)

    def check(self, test: CUDATestCase, fast: str, slow: str):
        test.assertTrue(all(instr in fast for instr in self.fast_expected))
        test.assertTrue(all(instr not in fast for instr in self.fast_unexpected))
        test.assertTrue(all(instr in slow for instr in self.slow_expected))
        test.assertTrue(all(instr not in slow for instr in self.slow_unexpected))

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


    def _test_fast_math_common(self, pyfunc, sig, device, criterion):
        
        # Test jit code path
        fastver = cuda.jit(sig, device=device, fastmath=True)(pyfunc)
        slowver = cuda.jit(sig, device=device)(pyfunc)
        # Test compile_ptx code path
        fastptx = compile_ptx_for_current_device(pyfunc, sig, device=device, fastmath=True)
        slowptx = compile_ptx_for_current_device(pyfunc, sig, device=device)
        criterion.check(self, fastver.ptx[sig], slowver.ptx[sig])

    def _test_fast_math_unary(self, op, criterion: FastMathCriterion):
        def kernel(r, x):
            r[0] = op(x)
        
        def device(x):
            return op(x)

        self._test_fast_math_common(kernel, (float32[::1], float32), device=False, criterion=criterion)
        self._test_fast_math_common(device, (float32,), device=True, criterion=criterion)
    

    def test_cosf(self):
        self._test_fast_math_unary(
            cos,
            FastMathCriterion(
                fast_expected=['cos.approx.ftz.f32 '],
                slow_unexpected=['cos.approx.ftz.f32 ']
            )
        )

    def test_sinf(self):
        self._test_fast_math_unary(
            sin,
            FastMathCriterion(
                fast_expected=['sin.approx.ftz.f32 '],
                slow_unexpected=['sin.approx.ftz.f32 ']
            )
        )

    def test_tanf(self):
        self._test_fast_math_unary(
            tan, 
            FastMathCriterion(fast_expected=[
                'sin.approx.ftz.f32 ',
                'cos.approx.ftz.f32 ',
                'div.approx.ftz.f32 '
            ], slow_unexpected=['sin.approx.ftz.f32 '])
        )

    def test_expf(self):
        self._test_fast_math_unary(
            exp, 
            FastMathCriterion(
                fast_unexpected=['fma.rn.f32 '],
                slow_expected=['fma.rn.f32 ']
            )
        )

    def test_logf(self):
        # Look for constant used to convert from log base 2 to log base e
        self._test_fast_math_unary(
            log, FastMathCriterion(
                fast_expected=['lg2.approx.ftz.f32 ', '0f3F317218'],
                slow_unexpected=['lg2.approx.ftz.f32 '],
            )
        )
    
    def test_log10f(self):
        # Look for constant used to convert from log base 2 to log base 10
        self._test_fast_math_unary(
            log10, FastMathCriterion(
                fast_expected=['lg2.approx.ftz.f32 ', '0f3E9A209B'],
                slow_unexpected=['lg2.approx.ftz.f32 ']
            )
        )

    def test_log2f(self):
        self._test_fast_math_unary(
            log2, FastMathCriterion(
                fast_expected=['lg2.approx.ftz.f32 '],
                slow_unexpected=['lg2.approx.ftz.f32 ']
            )
        )

    # def test_powf(self):
    #     self._test_fast_math(pow, ['lg2.approx.ftz.f32 '])

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


if __name__ == '__main__':
    unittest.main()
