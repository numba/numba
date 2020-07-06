#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy
import dppy as ocldrv


@dppy.kernel
def dppy_gemm(a, b, c):
    i = dppy.get_global_id(0)
    j = dppy.get_global_id(1)
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


def driver(device_env, a, b, c):
    # Copy the data to the device
    dA = device_env.copy_array_to_device(a)
    dB = device_env.copy_array_to_device(b)
    dC = device_env.create_device_array(c)
    # Invoke the kernel
    dppy_gemm[griddim,blockdim](dA, dB, dC)
    # Copy results back to host
    device_env.copy_array_from_device(dC)


def main():
    a = np.arange(X*X, dtype=np.float32).reshape(X,X)
    b = np.array(np.random.random(X*X), dtype=np.float32).reshape(X,X)
    c = np.ones_like(a).reshape(X,X)

    if ocldrv.has_gpu_device:
        with ocldrv.igpu_context(0) as device_env:
            driver(device_env, a, b, c)
    elif ocldrv.has_cpu_device:
        with ocldrv.cpu_context(0) as device_env:
            driver(device_env, a, b, c)
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
