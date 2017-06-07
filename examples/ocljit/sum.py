#! /usr/bin/env python
from __future__ import print_function

from timeit import default_timer as time

import numpy as np

from numba import ocl


@ocl.jit('(f4[:], f4[:], f4[:])')
def ocl_sum(a, b, c):
    i = ocl.get_global_id(1)
    c[i] = a[i] + b[i]

global_size = 50, 1
local_size = 32, 1, 1
N = global_size[0] * local_size[0]
print("N", N)
ocl_sum_configured = ocl_sum.configure(global_size, local_size)
a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.empty_like(a)

ts = time()
ocl_sum_configured(a, b, c)
te = time()
print(te - ts)
assert (a + b == c).all()
#print c
