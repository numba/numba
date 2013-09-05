# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numba import *

@jit(bool_(float64))
def isnan(x):
    return x != x

assert isnan(float('nan'))
assert not isnan(10.0)