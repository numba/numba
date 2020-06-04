import sys
import numpy as np
from numba import dppy
import math

import dppy.core as ocldrv

@dppy.func
def g(a):
    return a + 1

@dppy.kernel
def f(a, b):
    i = dppy.get_global_id(0)
    b[i] = g(a[i])

N = 10
a = np.ones(N)
b = np.ones(N)

device_env = None

try:
    device_env = ocldrv.runtime.get_gpu_device()
    print("Selected GPU device")
except:
    print("GPU device not found")
    exit()


# Copy the data to the device
dA = device_env.copy_array_to_device(a)
dB = device_env.copy_array_to_device(b)


print(b)
print("--------")
f[device_env, N](dA, dB)
device_env.copy_array_from_device(dB)
print(b)
