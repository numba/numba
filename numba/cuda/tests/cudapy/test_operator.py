import numpy as np
from numba.cuda.testing import (unittest, CUDATestCase, skip_unless_cc_53,
                                skip_on_cudasim)
from numba import cuda
from numba.core.types import f2,b1
import operator
from numba.cuda import compile_ptx


def simple_fp16_gt(ary, a, b):
    ary[0] = a > b


def simple_fp16_ge(ary, a, b):
    ary[0] = a >= b


def simple_fp16_lt(ary, a, b):
    ary[0] = a < b


def simple_fp16_le(ary, a, b):
    ary[0] = a <= b


def simple_fp16_eq(ary, a, b):
    ary[0] = a == b


def simple_fp16_ne(ary, a, b):
    ary[0] = a != b


class TestOperatorModule(CUDATestCase):
    """
    Test if operator module is supported by the CUDA target.
    """
    def operator_template(self, op):
        @cuda.jit
        def foo(a, b):
            i = 0
            a[i] = op(a[i], b[i])

        a = np.ones(1)
        b = np.ones(1)
        res = a.copy()
        foo[1, 1](res, b)

        np.testing.assert_equal(res, op(a, b))

    def test_add(self):
        self.operator_template(operator.add)

    def test_sub(self):
        self.operator_template(operator.sub)

    def test_mul(self):
        self.operator_template(operator.mul)

    def test_truediv(self):
        self.operator_template(operator.truediv)

    def test_floordiv(self):
        self.operator_template(operator.floordiv)

    @skip_unless_cc_53
    def test_fp16_comparison(self):
        functions = (simple_fp16_gt, simple_fp16_ge,
                     simple_fp16_lt, simple_fp16_le,
                     simple_fp16_eq, simple_fp16_ne)
        ops = (operator.gt, operator.ge, operator.lt, operator.le,
               operator.eq, operator.ne)

        for fn, op in zip(functions, ops):
            with self.subTest(op=op):
                kernel = cuda.jit("void(b1[:], f2, f2)")(fn)

                expected = np.zeros(1, dtype=np.bool8)
                got = np.zeros(1, dtype=np.bool8)
                arg1 = np.random.random(1).astype(np.float16)
                arg2 = np.random.random(1).astype(np.float16)

                kernel[1, 1](got, arg1[0], arg2[0])
                expected = op(arg1, arg2)
                self.assertEqual(got[0], expected)

    @skip_on_cudasim('Compilation unsupported in the simulator')
    def test_fp16_comparison_ptx(self):
        functions = (simple_fp16_gt, simple_fp16_ge,
                     simple_fp16_lt, simple_fp16_le,
                     simple_fp16_eq, simple_fp16_ne)
        ops = (operator.gt, operator.ge, operator.lt, operator.le,
               operator.eq, operator.ne)
        opstring = ('setp.gt.f16', 'setp.ge.f16',
                    'setp.lt.f16', 'setp.le.f16',
                    'setp.eq.f16', 'setp.ne.f16')
        args = (b1[:], f2, f2)

        for fn, op, s in zip(functions, ops, opstring):
            with self.subTest(op=op):
                ptx, _ = compile_ptx(fn, args, cc=(5, 3))
                self.assertIn(s, ptx)


if __name__ == '__main__':
    unittest.main()
