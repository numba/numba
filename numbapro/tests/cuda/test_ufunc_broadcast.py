from numbapro import cuda, vectorize
from numba import *
import numpy as np
import support
import unittest

@vectorize([f8(f8, f4, f4)], target='gpu')
def ufunc_broadcast(a, x, y):
    return a * x + y

class TestUFuncBroadcast(support.CudaTestCase):
    def test_ufunc_broadcast(self):
        N = 32
        X = np.arange(N, dtype=np.float32)
        Y = (np.arange(N, dtype=np.float32) + 1) * 2
        A = 2.0
        Z = ufunc_broadcast(A, X, Y)
        Z0 = A * X + Y
        self.assertTrue(np.allclose(Z, Z0))

if __name__ == '__main__':
    unittest.main()