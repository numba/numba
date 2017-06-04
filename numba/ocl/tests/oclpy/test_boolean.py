from __future__ import print_function, absolute_import
import numpy as np
from numba.ocl.testing import unittest
from numba import ocl


def boolean_func(A, vertial):
    if vertial:
        A[0] = 123
    else:
        A[0] = 321


class TestOclBoolean(unittest.TestCase):
    def test_boolean(self):
        func = ocl.jit('void(float64[:], bool_)')(boolean_func)
        A = np.array([0], dtype='float64')
        func(A, True)
        self.assertTrue(A[0] == 123)
        func(A, False)
        self.assertTrue(A[0] == 321)


if __name__ == '__main__':
    unittest.main()
