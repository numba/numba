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

# Array dimesnions
X = 8
Y = 8
global_size = X,Y

#a = np.array(np.random.random(X*Y), dtype=np.float32).reshape(X,Y)
#a.fill(1)
a = np.arange(X*Y, dtype=np.float32).reshape(X,Y)
b = np.array(np.random.random(X*Y), dtype=np.float32).reshape(X,Y)
#b.fill(1)
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

print("before : ", dB._ndarray)
print("before : ", dA._ndarray)
print("before : ", dC._ndarray)
data_parallel_sum[device_env,global_size](dA, dB, dC)
device_env.copy_array_from_device(dC)
print("after : ", dC._ndarray)

print("Done...")
