# -*- coding: utf-8 -*-
# from __future__ import division, absolute_import

from math import log1p
from numba import *
from numba.vectorize import vectorize
import numpy as np

@jit(double(double))
def jit_log1p(x):
    return log1p(x)

x = 3.4
assert np.allclose([jit_log1p(x)], [jit_log1p.py_func(x)])

@vectorize([double(double)])
def vec_log1p(x):
    return log1p(x)

x = np.array([x])
assert np.allclose(vec_log1p(x), [jit_log1p.py_func(x)])
