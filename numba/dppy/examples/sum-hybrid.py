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
    c[i] = a[i] + b[i]

N = 50*32
global_size = N,
print("N", N)

# Select a device for executing the kernel
gpu_env = None
cpu_env = None

try:
    gpu_env = ocldrv.runtime.get_gpu_device()
    print("Found GPU device")
    gpu_env.dump()
except:
    print("No GPU device")

try:
    cpu_env = ocldrv.runtime.get_cpu_device()
    print("Selected CPU device")
    cpu_env.dump()
except:
    print("No CPU device")

if cpu_env is not None:
    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)
    dA = cpu_env.copy_array_to_device(a)
    dB = cpu_env.copy_array_to_device(b)
    dC = ocldrv.DeviceArray(cpu_env.get_env_ptr(), c)
    print("before : ", dC._ndarray)
    data_parallel_sum[cpu_env,global_size](dA, dB, dC)
    cpu_env.copy_array_from_device(dC)
    print("after : ", dC._ndarray)

if gpu_env is not None:
    aa = np.array(np.random.random(N), dtype=np.float32)
    bb = np.array(np.random.random(N), dtype=np.float32)
    cc = np.ones_like(aa)
    dAA = gpu_env.copy_array_to_device(aa)
    dBB = gpu_env.copy_array_to_device(bb)
    dCC = ocldrv.DeviceArray(gpu_env.get_env_ptr(), cc)
    print("before : ", dCC._ndarray)
    data_parallel_sum[gpu_env,global_size](dAA, dBB, dCC)
    gpu_env.copy_array_from_device(dCC)
    print("after : ", dCC._ndarray)

print("Done...")
