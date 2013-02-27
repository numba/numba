# -*- coding: utf-8 -*-
'''
NumbaPro CUDA implementation
'''
from __future__ import print_function, division, absolute_import
from numbapro import cuda
from numba import autojit, jit, f8, int32, void
import numpy as np
import pylab
from timeit import default_timer as timer

@cuda.jit(void(f8[:], f8[:], f8[:], f8[:], f8, f8))
def cu_compute_error(X, Y, Ex, Ey, theta_x, theta_y):
    # Compute error for each element and store in the shared-memory
    Exsm = cuda.shared.array((1024,), dtype=f8)
    Eysm = cuda.shared.array((1024,), dtype=f8)

    tid = cuda.threadIdx.x
    base = cuda.blockIdx.x * cuda.blockDim.x

    i = base + tid

    x = X[i]
    y = Y[i]

    predict = theta_x + theta_y * x
    Exsm[tid] = predict - y
    Eysm[tid] = (predict - y) * x

    # Sum-reduce errors in the shared-memory
    n = cuda.blockDim.x
    while n > 1:

        cuda.syncthreads()

        half = n // 2
        if tid < half:
            Exsm[tid] += Exsm[tid + half]
            Eysm[tid] += Eysm[tid + half]

        n = half


    if tid == 0: # First of a block?
        # Store result
        Ex[cuda.blockIdx.x] = Exsm[0]
        Ey[cuda.blockIdx.x] = Eysm[0]


def gradient_descent(X, Y, theta, alpha, num_iters):
    N = X.size
    NTID = 1024
    NBLK = N // NTID
    assert NBLK * NTID == N

    Ex = np.empty(NBLK, dtype=X.dtype)
    Ey = np.empty(NBLK, dtype=X.dtype)

    theta_x, theta_y = 0, 0

    # -----------------
    # GPU work

    dX = cuda.to_device(X)
    dY = cuda.to_device(Y)

    dEx = cuda.to_device(Ex, copy=False)
    dEy = cuda.to_device(Ey, copy=False)

    griddim = NBLK,
    blockdim = NTID,

    for _ in xrange(num_iters):
        cu_compute_error[griddim, blockdim](dX, dY, dEx, dEy, theta_x, theta_y)

        dEx.to_host()
        dEy.to_host()

        # -----------------
        # CPU work

        error_x = Ex.sum()
        error_y = Ey.sum()

        theta_x = theta_x - alpha * (1.0 / N) * error_x
        theta_y = theta_y - alpha * (1.0 / N) * error_y
        
    theta[0] = theta_x
    theta[1] = theta_y


def populate_data(N):
    noise = np.random.random(N).astype(np.float64)
    X = np.arange(N, dtype=np.float64)
    slope = 3
    Y = noise * (slope * X)
    return X, Y


def plot(X, theta, c='r'):
    result = theta[0] + theta[1] * X
    pylab.plot(X, result, c=c, linewidth=2)


def main():
    NBLK = 10
    NTID = 1024
    N = NBLK * NTID
    print 'N = %d' % N

    X, Y = populate_data(N)

    theta = np.zeros(2, dtype=X.dtype)

    ts = timer()
    gradient_descent(X, Y, theta, 1e-10, 1000)
    te = timer()
    timing = te - ts

    print 'Time elapsed: %s' % timing


    pylab.scatter(X, Y, marker='o', c='b')
    pylab.title('Linear Regression')
    plot(X, theta)
    pylab.show()


if __name__ == '__main__':
    main()
