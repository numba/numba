from numba.decorators import jit
from numba import *

@jit(arg_types=[d[:], d[:], d[:]])
def add_d(A, B, C):
    N = A.shape[0]
    for i in range(N):
        C[i] = A[i] + B[i]


