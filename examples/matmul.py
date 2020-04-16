#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy
import dppy.core as ocldrv


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

a = np.arange(X*X, dtype=np.float32).reshape(X,X)
b = np.array(np.random.random(X*X), dtype=np.float32).reshape(X,X)
c = np.ones_like(a).reshape(X,X)

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
# Invoke the kernel
dppy_gemm[device_env,griddim,blockdim](dA, dB, dC)
# Copy results back to host
device_env.copy_array_from_device(dC)

# Host compute using standard Numpy
Amat = np.matrix(a)
Bmat = np.matrix(b)
Cans = Amat * Bmat

# Check result
assert np.allclose(c, Cans)
print("Done...")
