# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys
from numba import *
import numpy as np
import numba
@jit(object_(double[:, :]))
def func2(A):
    L = []
    n = A.shape[0]

    for i in range(10):
        for j in range(10):
            temp = A[i-n : i+n+1, j-2 : j+n+1]
            L.append(temp)

    return L

A = np.empty(1000000, dtype=np.double).reshape(1000, 1000)

refcnt = sys.getrefcount(A)
func2(A)
new_refcnt = sys.getrefcount(A)
assert refcnt == new_refcnt

normal_result = list(map(sys.getrefcount, func2.py_func(A)))
numba_result = list(map(sys.getrefcount, func2(A)))
assert normal_result == numba_result
