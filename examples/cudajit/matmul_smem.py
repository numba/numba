from numbapro import cuda
from numba import *
import numpy as np
import math
from timeit import default_timer as time

bpg = 10
tpb = 20
n = bpg * tpb

@cuda.jit(argtypes=[f4[:,:], f4[:,:], f4[:,:]])
def cu_square_matrix_mul(A, B, C):
    sA = cuda.shared.array(shape=(tpb, n), dtype=f4)
    sB = cuda.shared.array(shape=(n, tpb), dtype=f4)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    x = tx + bx * bw
    y = ty + by * bh

    if x < n or y < n:
        for i in range(bpg):
            sA[ty, tx + i * tpb] = A[y, tx + i * tpb]
            sB[ty + i * tpb, tx] = B[ty + i * tpb, x]


    cuda.syncthreads()

    if x >= n or y >= n:
        return # no thread can die before a syncthread

    C[y, x] = 0
    for i in range(n):
        C[y, x] += sA[ty, i] * sB[i, tx]


A = np.array(np.random.random((n, n)), dtype=np.float32)
B = np.array(np.random.random((n, n)), dtype=np.float32)
C = np.empty_like(A)

print "N = %d x %d" % (n, n)

s = time()
cu_square_matrix_mul[(bpg, bpg), (tpb, tpb)](A, B, C)
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


