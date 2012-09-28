#! /usr/bin/env python
# ______________________________________________________________________
'''test_forloop

Test the Numba compiler on a simple for loop over an iterable object.
'''
# ______________________________________________________________________

import unittest

import numba
from numba import *
from numba.decorators import autojit

import numpy as np

@autojit(backend='bytecode')
def matmulcore(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]


class TestASTArrays(unittest.TestCase):

    def test_numba(self):
        A = np.arange(16, dtype=np.float32).reshape(4, 4)
        B = np.arange(16, dtype=np.float32).reshape(4, 4)
        C = np.zeros(16, dtype=np.float32).reshape(4, 4)
        Gold = np.matrix(A) * np.matrix(B)

        matmulcore(A, B, C)

        self.assertTrue(np.all(C == Gold))

if __name__ == '__main__':
    unittest.main()
