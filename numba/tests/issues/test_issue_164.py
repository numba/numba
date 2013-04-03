# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import numpy as np
from numba import jit, double

def expr_py(a, b, c):
    length = len(a)
    result = np.empty(length, dtype=np.float64)
    for i in range(length):
        result[i] = b[i] ** 2 if a[i] > 0.1 else c[i] ** 3
    return result
expr_nb = jit(double[:](double[:], double[:], double[:]))(expr_py)

size = 1e6

a = np.random.rand(size)
b = np.random.rand(size)
c = np.random.rand(size)

assert np.allclose(expr_nb(a, b, c), expr_py(a, b, c))
