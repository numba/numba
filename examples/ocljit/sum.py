#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import numpy as np
from numba import ocl


@ocl.jit
def ocl_sum(a, b, c):
    i = ocl.get_global_id(0)
    c[i] = a[i] + b[i]


global_size = 50, 1
local_size = 32, 1, 1
N = global_size[0] * local_size[0]
print("N", N)

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.ones_like(a)

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
