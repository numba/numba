#! /usr/bin/env python
from timeit import default_timer as time
import numpy as np
from numba import ocl


@ocl.jit
def matmult(A, B, C):
    x = ocl.get_global_id(1)
    y = ocl.get_global_id(0)
    if x >= C.shape[0] or y >= C.shape[1]:
        return
    C[y, x] = 0
    for i in range(C.shape[0]):
        C[y, x] += A[y, i] * B[i, x]


blocksize = 64
gridsize = 16
N = gridsize * blocksize

A = np.array(np.random.random((N, N)), dtype=np.float32)
B = np.array(np.random.random((N, N)), dtype=np.float32)
C = np.zeros_like(A)

griddim = N, N
blockdim = blocksize, blocksize

ts = time()

dA = ocl.to_device(A)
dB = ocl.to_device(B)
dC = ocl.device_array_like(C)
matmult[griddim, blockdim](dA, dB, dC)#(A,B,C)
dC.copy_to_host(C)

te = time()
tocl = te - ts

# Host compute
Amat = np.matrix(A)
Bmat = np.matrix(B)

ts = time()
Cans = Amat * Bmat
te = time()
tcpu = te - ts

# Check result
assert np.allclose(C, Cans)

print('cpu:  %f' % tcpu)
print('ocl: %f' % tocl)
print('ocl speedup: %.2fx' % (tcpu / tocl))

