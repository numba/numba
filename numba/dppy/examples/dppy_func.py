import sys
import numpy as np
from numba import dppy
import math

import dppy as ocldrv


@dppy.func
def g(a):
    return a + 1


@dppy.kernel
def f(a, b):
    i = dppy.get_global_id(0)
    b[i] = g(a[i])


def driver(device_env, a, b, N):
    # Copy the data to the device
    dA = device_env.copy_array_to_device(a)
    dB = device_env.copy_array_to_device(b)

    print(b)
    print("--------")
    f[N, dppy.DEFAULT_LOCAL_SIZE](dA, dB)
    device_env.copy_array_from_device(dB)
    print(b)


def main():
    N = 10
    a = np.ones(N)
    b = np.ones(N)

    if ocldrv.has_gpu_device:
        with ocldrv.igpu_context(0) as device_env:
            driver(device_env, a, b, N)
    elif ocldrv.has_cpu_device:
        with ocldrv.cpu_context(0) as device_env:
            driver(device_env, a, b, N)
    else:
        print("No device found")


if __name__ == '__main__':
    main()
