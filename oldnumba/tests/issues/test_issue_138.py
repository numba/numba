# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import numpy as np
from numba import autojit, jit, double, void, int32

@autojit
def limiter(x, n):
    if x + 1 > n:
        return n
    else:
        return x + 1

@jit(void(double[:,:,:], double[:,:,:], double[:,:,:], int32, int32, int32))
def gather_convolution_core(A, B, C, x, y, z):
    dimA = A.shape[0]
    bx = limiter(x, dimA)
    by = limiter(y, dimA)
    bz = limiter(z, dimA)
    for x1 in range(bx):
        for x2 in range(bx):
            if x1 + x2 == x:
                for y1 in range(by):
                    for y2 in range(by):
                        if y1 + y2 == y:
                            for z1 in range(bz):
                                for z2 in range(bz):
                                    if z1 + z2 == z:
                                        C[x, y, z] += A[x1, y1, z1] * B[x2, y2, z2]


def gather_convolution(A, B, C):
    from itertools import product
    dimC = C.shape[0]
    for x, y, z in product(range(dimC), range(dimC), range(dimC)):
        gather_convolution_core(A, B, C, x, y, z)

def test_convolution():
    # Creating some fake data to test this problem
    s = 4
    array_a = np.random.rand(s ** 3).reshape(s, s, s)
    array_b = np.random.rand(s ** 3).reshape(s, s, s)

    dimA = array_a.shape[0]
    dimB = array_b.shape[0]
    dimC = dimA + dimB

    array_c = np.zeros((dimC, dimC, dimC))

    gather_convolution(array_a, array_b, array_c)

if __name__ == '__main__':
    test_convolution()

