#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy
from numba.dppy.dppy_driver import driver as ocldrv


@dppy.kernel
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]

global_size = 10
N = global_size
print("N", N)


a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.ones_like(a)

# Select a device for executing the kernel
device_env = None

try:
    device_env = ocldrv.runtime.get_gpu_device()
    print("Selected GPU device")
except:
    try:
        device_env = ocldrv.runtime.get_cpu_device()
        print("Selected CPU device")
    except:
        print("No OpenCL devices found on the system")
        raise SystemExit()


# Copy the data to the device
dA = device_env.copy_array_to_device(a)
dB = device_env.copy_array_to_device(b)
dC = ocldrv.DeviceArray(device_env.get_env_ptr(), c)

print("before : ", dA._ndarray)
print("before : ", dB._ndarray)
print("before : ", dC._ndarray)
data_parallel_sum[device_env,global_size](dA, dB, dC)
device_env.copy_array_from_device(dC)
print("after : ", dC._ndarray)

print("Done...")
