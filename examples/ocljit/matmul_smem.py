#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import numpy as np
from numba import ocl, float32


bpg = 50
tpb = 32
n = bpg * tpb


@ocl.jit('(float32[:,:], float32[:,:], float32[:,:])')
def cl_square_matrix_mul(A, B, C):
    sA = ocl.shared.array(shape=(tpb, tpb), dtype=float32)
    sB = ocl.shared.array(shape=(tpb, tpb), dtype=float32)

    tx = ocl.get_local_id(0)
    ty = ocl.get_local_id(1)

    x = ocl.get_global_id(0)
    y = ocl.get_global_id(1)

    acc = 0.
    for i in range(bpg):
        if x < n and y < n:
            sA[ty, tx] = A[y, tx + i * tpb]
            sB[ty, tx] = B[ty + i * tpb, x]

        ocl.barrier()

        if x < n and y < n:
            for j in range(tpb):
                acc += sA[ty, j] * sB[j, tx]

        ocl.barrier()

    if x < n and y < n:
        C[y, x] = acc

A = np.array(np.random.random((n, n)), dtype=np.float32)
B = np.array(np.random.random((n, n)), dtype=np.float32)
C = np.empty_like(A)

print("N = %d x %d" % (n, n))

s = time()
#dA = ocl.to_device(A)
#dB = ocl.to_device(B)
#dC = ocl.device_array_like(C)
cl_square_matrix_mul[(bpg, bpg), (tpb, tpb)](A, B, C)
#dC.copy_to_host(C)

e = time()
tocl = e - s

# Host compute
Amat = np.matrix(A)
Bmat = np.matrix(B)

s = time()
Cans = Amat * Bmat
e = time()
tcpu = e - s

print('cpu:  %f' % tcpu)
print('ocl: %f' % tocl)
print('ocl speedup: %.2fx' % (tcpu / tocl))

# Check result
assert np.allclose(C, Cans)
