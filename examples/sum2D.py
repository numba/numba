#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy
from numba.dppy.dppy_driver import driver as ocldrv


@dppy.jit
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    j = dppy.get_global_id(1)
    c[i,j] = a[i,j] + b[i,j]

global_size = 48, 1
local_size = 3, 1, 1

# Array dimesnions
X = 1024
Y = 1024

a = np.array(np.random.random(X*Y), dtype=np.float32).reshape(X,Y)
b = np.array(np.random.random(X*Y), dtype=np.float32).reshape(X,Y)
c = np.ones_like(a).reshape(X,Y)

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

print("before : ", dC._ndarray)
#data_parallel_sum[device_env,global_size,local_size](dA, dB, dC)
device_env.copy_array_from_device(dC)
print("after : ", dC._ndarray)

print("Done...")
