#! /usr/bin/env python
from timeit import default_timer as time
import numpy as np
from numba import ocl


blocksize = 20
gridsize = 20


@ocl.jit
def matmult(A, B, C):
    x = ocl.get_global_id(0)
    y = ocl.get_global_id(1)
    if x >= C.shape[0] or y >= C.shape[1]:
        return
    C[y, x] = 0
    for i in range(gridsize):
        C[y, x] += A[y, i] * B[i, x]


N = gridsize * blocksize
A = np.array(np.random.random((N, N)), dtype=np.float32)
B = np.array(np.random.random((N, N)), dtype=np.float32)
C = np.zeros_like(A)

griddim = gridsize, gridsize
blockdim = blocksize, blocksize

ts = time()

#dA = ocl.to_device(A)
#dB = ocl.to_device(B)
#dC = ocl.to_device(C)
matmult[griddim, blockdim](A,B,C)#(dA, dB, dC)
#dC.to_host(stream)

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

