from numbapro import cuda
from numba import *
import numpy as np
import math
from time import time

@cuda.jit(argtypes=[f4[:,:], f4[:,:], f4[:,:]], target='gpu')
def cu_square_matrix_mul(A, B, C):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    x = tx + bx * bw
    y = ty + by * bh

    n = C.shape[0]
    
    if x >= n:
        return
    if y >= n:
        return

    C[y, x] = 0
    for i in range(n):
        C[y, x] += A[y, i] * B[i, x]

n = 1000

A = np.array(np.random.random((n, n)), dtype=np.float32)
B = np.array(np.random.random((n, n)), dtype=np.float32)
C = np.empty_like(A)

# Device compute
blkn = int(math.ceil(n / 32.))

s = time()
cu_square_matrix_mul[(blkn, blkn), (32, 32)](A, B, C)
e = time()
tcuda = e - s

# Host compute
Amat = np.matrix(A)
Bmat = np.matrix(B)

s = time()
Cans = Amat * Bmat
e = time()
tcpu = e - s

# Check result
relerr = lambda got, gold: abs(got - gold)/gold
for y in range(n):
    for x in range(n):
        err = relerr(C[y, x], Cans[y, x])
        assert err < 1e-5, (x, y, err)

print 'cpu:  %f' % tcpu
print 'cuda: %f' % tcuda
print 'cuda speedup: %.2fx' % (tcpu / tcuda)


