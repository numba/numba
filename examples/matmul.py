#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppl
import dpctl


@dppl.kernel
def dppl_gemm(a, b, c):
    i = dppl.get_global_id(0)
    j = dppl.get_global_id(1)
    if i >= c.shape[0] or j >= c.shape[1]:
        return
    c[i,j] = 0
    for k in range(c.shape[0]):
        c[i, j] += a[i, k] * b[k, j]


# Array dimesnions
X = 1024
Y = 16
global_size = X,X

griddim = X, X
blockdim = Y, Y


def driver(a, b, c):
    # Invoke the kernel
    dppl_gemm[griddim,blockdim](a, b, c)


def main():
    a = np.arange(X*X, dtype=np.float32).reshape(X,X)
    b = np.array(np.random.random(X*X), dtype=np.float32).reshape(X,X)
    c = np.ones_like(a).reshape(X,X)

    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            driver(a, b, c)
    elif dpctl.has_cpu_queues():
        with dpctl.device_context("opencl:cpu") as cpu_queue:
            driver(a, b, c)
    else:
        print("No device found")
        exit()

    # Host compute using standard Numpy
    Amat = np.matrix(a)
    Bmat = np.matrix(b)
    Cans = Amat * Bmat

    # Check result
    assert np.allclose(c, Cans)
    print("Done...")


if __name__ == '__main__':
    main()
