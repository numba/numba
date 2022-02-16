import numpy as np
from numba.cuda.testing import unittest, CUDATestCase, skip_unless_cc_53
from numba import cuda
import operator


def simple_fp16add(ary, a, b):
    ary[0] = a + b


def simple_fp16sub(ary, a, b):
    ary[0] = a - b


def simple_fp16mul(ary, a, b):
    ary[0] = a * b


def simple_fp16neg(ary, a):
    ary[0] = -a


def simple_fp16abs(ary, a):
    ary[0] = abs(a)


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
    def test_fp16_binary(self):
        functions = (simple_fp16add, simple_fp16sub, simple_fp16mul)
        ops = (operator.add, operator.sub, operator.mul)

        for fn, op in zip(functions, ops):
            with self.subTest(op=op):
                kernel = cuda.jit("void(f2[:], f2, f2)")(fn)

                expected = np.zeros(1, dtype=np.float16)
                got = np.zeros(1, dtype=np.float16)
                arg1 = np.random.random(1).astype(np.float16)
                arg2 = np.random.random(1).astype(np.float16)

                kernel[1, 1](got, arg1[0], arg2[0])
                expected = op(arg1, arg2)
                np.testing.assert_allclose(got[0], expected)

    @skip_unless_cc_53
    def test_fp16_unary(self):
        functions = (simple_fp16neg, simple_fp16abs)
        ops = (operator.neg, operator.abs)

        for fn, op in zip(functions, ops):
            with self.subTest(op=op):
                kernel = cuda.jit("void(f2[:], f2)")(fn)

                expected = np.zeros(1, dtype=np.float16)
                got = np.zeros(1, dtype=np.float16)
                arg1 = np.random.random(1).astype(np.float16)

                kernel[1, 1](got, arg1[0])
                expected = op(arg1)
                np.testing.assert_allclose(got[0], expected)


if __name__ == '__main__':
    unittest.main()
