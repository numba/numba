import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba import cuda
import operator


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


if __name__ == '__main__':
    unittest.main()
