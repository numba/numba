from numbapro import guvectorize
from numba import *
import math
import numpy as np
import numpy.core.umath_tests as ut
import unittest

def matmulcore(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]


class TestVectorizeDecor(unittest.TestCase):

    def test_cuda_guvectorize(self):
        target = 'gpu'

        gufunc = guvectorize([void(f4[:,:], f4[:,:], f4[:,:])],
                             '(m,n),(n,p)->(m,p)',
                             target=target)(matmulcore)
        gufunc.max_blocksize = 512
        matrix_ct = 1001 # an odd number to test thread/block division in CUDA
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)

        C = gufunc(A, B)
        Gold = ut.matrix_multiply(A, B)

        self.assertTrue(np.allclose(C, Gold))


if __name__ == '__main__':
    unittest.main()