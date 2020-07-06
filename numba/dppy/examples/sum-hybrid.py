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


N = 50*32
global_size = N,


def main():
    if ocldrv.has_cpu_device:
        with ocldrv.cpu_context(0) as device_env:
            print("-----Running in CPU-----")
            a = np.array(np.random.random(N), dtype=np.float32)
            b = np.array(np.random.random(N), dtype=np.float32)
            c = np.ones_like(a)
            dA = device_env.copy_array_to_device(a)
            dB = device_env.copy_array_to_device(b)
            dC = device_env.create_device_array(c)
            print("before A: ", dA.get_ndarray())
            print("before B: ", dB.get_ndarray())
            data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](dA, dB, dC)
            device_env.copy_array_from_device(dC)
            print("after  C: ", dC.get_ndarray())
    else:
        print("CPU device not found")

    if ocldrv.has_gpu_device:
        with ocldrv.igpu_context(0) as device_env:
            print("-----Running in GPU-----")
            a = np.array(np.random.random(N), dtype=np.float32)
            b = np.array(np.random.random(N), dtype=np.float32)
            c = np.ones_like(a)
            dA = device_env.copy_array_to_device(a)
            dB = device_env.copy_array_to_device(b)
            dC = device_env.create_device_array(c)
            print("before A: ", dA.get_ndarray())
            print("before B: ", dB.get_ndarray())
            data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](dA, dB, dC)
            device_env.copy_array_from_device(dC)
            print("after  C: ", dC.get_ndarray())
    else:
        print("GPU device not found")
        exit()

    print("Done...")


if __name__ == '__main__':
    main()
