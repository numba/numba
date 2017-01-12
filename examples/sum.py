#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import double
from numba.decorators import jit as jit

def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

jitsum2d = jit(sum2d)
csum2d = jitsum2d.compile(double(double[:,::1]))

from numpy import random
arr = random.randn(100, 100)

import time
start = time.time()
res = sum2d(arr)
duration = time.time() - start
print("Result from python is %s in %s (msec)" % (res, duration*1000))

csum2d(arr)       # warm up

start = time.time()
res = csum2d(arr)
duration2 = time.time() - start
print("Result from compiled is %s in %s (msec)" % (res, duration2*1000))

print("Speed up is %s" % (duration / duration2))
