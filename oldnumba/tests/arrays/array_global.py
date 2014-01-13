from numba import *
import numpy as np

A = np.arange(10, dtype=np.float32)
X = np.empty(10, dtype=np.float32)

@jit(void(f4[:]))
def read_global(Y):
    Y[1] = A[1]

read_global(X)
assert X[1] == 1

@jit(void(f4[:]))
def write_global(Y):
    A[2] = Y[2]

X[2] = 14
write_global(X)
assert A[2] == 14
