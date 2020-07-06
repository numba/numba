#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy
import dppy.core as ocldrv


@dppy.kernel
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    j = dppy.get_global_id(1)
    c[i,j] = a[i,j] + b[i,j]


def driver(device_env, a, b, c, global_size):
    # Copy the data to the device
    dA = device_env.copy_array_to_device(a)
    dB = device_env.copy_array_to_device(b)
    dC = device_env.create_device_array(c)

    print("before A: ", dA.get_ndarray())
    print("before B: ", dB.get_ndarray())
    data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](dA, dB, dC)
    device_env.copy_array_from_device(dC)
    print("after  C : ", dC.get_ndarray())


def main():
    # Array dimesnions
    X = 8
    Y = 8
    global_size = X,Y

    a = np.arange(X*Y, dtype=np.float32).reshape(X,Y)
    b = np.array(np.random.random(X*Y), dtype=np.float32).reshape(X,Y)
    c = np.ones_like(a).reshape(X,Y)

    if ocldrv.has_gpu_device:
        with ocldrv.igpu_context(0) as device_env:
            driver(device_env, a, b, c, global_size)
    elif ocldrv.has_cpu_device:
        with ocldrv.cpu_context(0) as device_env:
            driver(device_env, a, b, c, global_size)
    else:
        print("No device found")
        exit()

    print("Done...")


if __name__ == '__main__':
    main()
