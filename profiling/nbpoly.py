from numba.decorators import jit
from numba import *
from math import sqrt

@jit(argtypes=[double[:], double[:], double[:], double[:]])
def poly_d(A, B, C, D):
    N = A.shape[0]
    for i in range(N):
        D[i] = sqrt(B[i]**2 + 4 * A[i] * C[i])


