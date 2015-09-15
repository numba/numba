from __future__ import absolute_import, print_function, division
from numba import unittest_support as unittest
from numba import guvectorize
from numba import void, float32
import numpy as np
import numpy.core.umath_tests as ut


def matmulcore(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]


class TestVectorizeDecor(unittest.TestCase):

    def test_cpu_guvectorize(self):
        target = 'cpu'

        gufunc = guvectorize([void(float32[:,:], float32[:,:], float32[:,:])],
                             '(m,n),(n,p)->(m,p)',
                             target=target)(matmulcore)

        matrix_ct = 1001 # an odd number to test thread/block division in CUDA
        A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
        B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)

        C = gufunc(A, B)
        Gold = ut.matrix_multiply(A, B)

        self.assertTrue(np.allclose(C, Gold))


if __name__ == '__main__':
    unittest.main()
