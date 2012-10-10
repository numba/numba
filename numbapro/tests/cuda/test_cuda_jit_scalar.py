import unittest
import numpy as np
from numba import *
from numbapro import cuda

def cu_scalar_dummy(x):
    pass


class TestCudaJitScalar(unittest.TestCase):
    def test_scalar_i4(self):
        cufunc = jit(argtypes=[i4])(cu_scalar_dummy)
        cufunc(100)

    def test_scalar_i8(self):
        cufunc = jit(argtypes=[i8])(cu_scalar_dummy)
        cufunc(100)

    def test_scalar_f4(self):
        cufunc = jit(argtypes=[f4])(cu_scalar_dummy)
        cufunc(100)

    def test_scalar_f8(self):
        cufunc = jit(argtypes=[f8])(cu_scalar_dummy)
        cufunc(100)

if __name__ == '__main__':
    unittest.main()


