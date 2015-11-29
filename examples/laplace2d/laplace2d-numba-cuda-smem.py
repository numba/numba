#!/usr/bin/env python

'''
Speed on OS X 10.8 650M 1024GB GPU: 186s
'''

import numpy as np
import time
from numba import *


# NOTE: CUDA kernel does not return any value

@cuda.jit(argtypes=[f8[:,:], f8[:,:], f8[:,:]])
def jacobi_relax_core(A, Anew, error):
    smem = cuda.shared.array(shape=(32 + 2, 32 + 2), dtype=f4)
    n = A.shape[0]
    m = A.shape[1]

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    j = ty + cuda.blockIdx.y * cuda.blockDim.y
    i = tx + cuda.blockIdx.x * cuda.blockDim.x
    
    sy = ty + 1
    sx = tx + 1
    
    smem[sy, sx] = A[j, i]
    if tx == 0 and i >= 1:
        smem[sy, 0] = A[j, i - 1]

    if ty == 0 and j < m - 1:
        smem[0, sx] = A[j - 1, i]

    if tx == 31 and j >= 1:
        smem[sy, 33] = A[j, i + 1]

    if ty == 31 and j < n - 1:
        smem[33, sx] = A[j + 1, i]

    cuda.syncthreads() # ensure smem is visible by all threads in the block
    
    if j >= 1 and j < n - 1 and i >= 1 and i < m - 1:
        Anew[j, i] = 0.25 * ( smem[sy, sx + 1] + smem[sy, sx - 1] \
                            + smem[sy - 1, sx] + smem[sy + 1, sx])
        error[j, i] = Anew[j, i] - A[j, i]

def main():
    NN = 512
    NM = 512

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

    blockdim = (32, 32)
    griddim = (NN//blockdim[0], NM//blockdim[1])
        
    error_grid = np.zeros_like(A)
    
    stream = cuda.stream()

    dA = cuda.to_device(A, stream)          # to device and don't come back
    dAnew = cuda.to_device(Anew, stream)    # to device and don't come back
    derror_grid = cuda.to_device(error_grid, stream)
    
    while error > tol and iter < iter_max:
        assert error_grid.dtype == np.float64
        
        jacobi_relax_core[griddim, blockdim, stream](dA, dAnew, derror_grid)
        
        derror_grid.to_host(stream)
        
        
        # error_grid is available on host
        stream.synchronize()
        
        error = np.abs(error_grid).max()
        
        # swap dA and dAnew
        tmp = dA
        dA = dAnew
        dAnew = tmp

        if iter % 100 == 0:
            print("%5d, %0.6f (elapsed: %f s)" % (iter, error, time.time()-timer))

        iter += 1

    runtime = time.time() - timer
    print(" total: %f s" % runtime)

if __name__ == '__main__':
    main()
