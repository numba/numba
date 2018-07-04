#!/usr/bin/env python

from __future__ import print_function

import time

import numpy as np

from numba import jit, prange, stencil

@stencil
def jacobi_kernel(A):
    return 0.25 * (A[0,1] + A[0,-1] + A[-1,0] + A[1,0])

@jit(parallel=True)
def jacobi_relax_core(A, Anew):
    error = 0.0
    n = A.shape[0]
    m = A.shape[1]
    Anew = jacobi_kernel(A)
    error = np.max(np.abs(Anew - A))
    return error


def main():
    NN = 3000
    NM = 3000

    A = np.zeros((NN, NM), dtype=np.float64)
    Anew = np.zeros((NN, NM), dtype=np.float64)

    n = NN
    m = NM
    iter_max = 1000

    tol = 1.0e-6
    error = 1.0

    for j in range(n):
        A[j, 0] = 1.0
        Anew[j, 0] = 1.0

    print("Jacobi relaxation Calculation: %d x %d mesh" % (n, m))

    timer = time.time()
    iter = 0

    while error > tol and iter < iter_max:
        error = jacobi_relax_core(A, Anew)

        # swap A and Anew
        tmp = A
        A = Anew
        Anew = tmp

        if iter % 100 == 0:
            print("%5d, %0.6f (elapsed: %f s)" % (iter, error, time.time()-timer))

        iter += 1

    runtime = time.time() - timer
    print(" total: %f s" % runtime)

if __name__ == '__main__':
    main()
