#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy
from numba.dppy.dppy_driver import driver as ocldrv


@dppy.kernel(access_types={"read_only": ['a', 'b'], "write_only": ['c'], "read_write": []})
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]

global_size = 64
local_size = 32
N = global_size * local_size
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

print("before : ", c)
data_parallel_sum[device_env,global_size,local_size](a, b, c)
print("after : ", c)

print("Done...")
