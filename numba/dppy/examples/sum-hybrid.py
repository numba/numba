#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import pdb
import sys
import numpy as np
from numba import dppy
from numba.dppy.dppy_driver import driver as ocldrv


@dppy.jit
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]

global_size = 50, 1
local_size = 32, 1, 1
N = global_size[0] * local_size[0]
print("N", N)




# Select a device for executing the kernel
gpu_env = None
cpu_env = None

try:
    gpu_env = ocldrv.runtime.get_gpu_device()
    print("Found GPU device")
except:
    print("No GPU device")

try:
    cpu_env = ocldrv.runtime.get_cpu_device()
    print("Selected CPU device")
except:
    print("No CPU device")

if cpu_env is not None:
    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)
    cpu_env.dump()
    dA = cpu_env.copy_array_to_device(a)
    dB = cpu_env.copy_array_to_device(b)
    dC = ocldrv.DeviceArray(cpu_env.get_env_ptr(), c)
    print("before : ", dC._ndarray)
    data_parallel_sum[cpu_env,global_size,local_size](dA, dB, dC)
    cpu_env.copy_array_from_device(dC)
    print("after : ", dC._ndarray)
'''
if cpu_env is not None:
    cpu_env.dump()
    dA = cpu_env.copy_array_to_device(a)
    dB = cpu_env.copy_array_to_device(b)
    dC = ocldrv.DeviceArray(cpu_env.get_env_ptr(), c)
    print("before : ", dC._ndarray)
    data_parallel_sum[cpu_env,global_size,local_size](dA, dB, dC)
    cpu_env.copy_array_from_device(dC)
    print("after : ", dC._ndarray)
'''
if gpu_env is not None:
    print("============================")
    aa = np.array(np.random.random(N), dtype=np.float32)
    bb = np.array(np.random.random(N), dtype=np.float32)
    cc = np.ones_like(aa)
    gpu_env.dump()
    dAA = gpu_env.copy_array_to_device(aa)
    dBB = gpu_env.copy_array_to_device(bb)
    dCC = ocldrv.DeviceArray(gpu_env.get_env_ptr(), cc)
    print("before : ", dCC._ndarray)
    pdb.set_trace()
    data_parallel_sum[gpu_env,global_size,local_size](dAA, dBB, dCC)
    gpu_env.copy_array_from_device(dCC)
    print("after : ", dCC._ndarray)

print("Done...")
