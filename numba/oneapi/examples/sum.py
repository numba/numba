#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import numpy as np
from numba import oneapi
from numba.oneapi.oneapidriver import driver as oneapidrv


@oneapi.jit
def data_parallel_sum(a, b, c):
    i = oneapi.get_global_id(0)
    c[i] = a[i] + b[i]


global_size = 50, 1
local_size = 32, 1, 1
N = global_size[0] * local_size[0]
print("N", N)


a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.ones_like(a)

# Select a device for executing the kernel
device_env = None

if oneapidrv.runtime.get_gpu_device() is not None:
    device_env = oneapidrv.runtime.get_gpu_device()
elif oneapidrv.runtime.get_cpu_device() is not None:
    device_env = oneapidrv.runtime.get_cpu_device()
else:
    e = oneapidrv.DeviceNotFoundError("No OpenCL devices found on the system")
    raise e

# Copy the data to the device
dA = device_env.copy_array_to_device(a)
dB = device_env.copy_array_to_device(b)
dC = oneapidrv.DeviceArray(device_env.get_env_ptr(), c)

data_parallel_sum[device_env,global_size,local_size](dA, dB, dC)

"""
ts = time()
#dA = ocl.to_device(a)
#dB = ocl.to_device(b)
#dC = ocl.device_array_like(c)
ocl_sum[global_size,local_size](a, b, c)
#dC.copy_to_host(c)
te = time()

print(te - ts)
assert (a + b == c).all()
#print c
"""
