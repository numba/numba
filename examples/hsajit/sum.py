#! /usr/bin/env python
from __future__ import print_function

from timeit import default_timer as time

import numpy as np

from numba import hsa


@hsa.jit('(f4[:], f4[:], f4[:])')
def hsa_sum(a, b, c):
    i = hsa.get_global_id(0)
    c[i] = a[i] + b[i]

griddim = 50, 1
blockdim = 32, 1, 1
N = griddim[0] * blockdim[0]
print("N", N)
hsa_sum_configured = hsa_sum.configure(griddim, blockdim)
a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.empty_like(a)

ts = time()
with hsa.register(a,b,c):
    hsa_sum_configured(a, b, c)
te = time()
print(te - ts)
assert (a + b == c).all()
#print c
