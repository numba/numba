#! /usr/bin/env python
# ______________________________________________________________________
'''test_forloop

Test the Numba compiler on a simple for loop over an iterable object.
'''
# ______________________________________________________________________

import numba
from numba import *
from numba.decorators import function

import numpy as np

@function
def matmulcore(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] = C[i, j] + A[i, k] * B[k, j]


def test_numba():
    A = np.arange(16, dtype=np.float32).reshape(4, 4)
    B = np.arange(16, dtype=np.float32).reshape(4, 4)
    C = np.zeros(16, dtype=np.float32).reshape(4, 4)
    Gold = np.matrix(A) * np.matrix(B)

    matmulcore(A, B, C)

    if (C != Gold).any():
        raise ValueError

if __name__ == '__main__':
    test_numba()