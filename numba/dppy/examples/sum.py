#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy
import dppy as ocldrv


@dppy.kernel
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


def driver(device_env, a, b, c, global_size):
    # Copy the data to the device
    dA = device_env.copy_array_to_device(a)
    dB = device_env.copy_array_to_device(b)
    dC = device_env.create_device_array(c)

    print("before : ", dA._ndarray)
    print("before : ", dB._ndarray)
    print("before : ", dC._ndarray)
    data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](dA, dB, dC)
    device_env.copy_array_from_device(dC)
    print("after : ", dC._ndarray)


def main():
    global_size = 10
    N = global_size
    print("N", N)

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)

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
