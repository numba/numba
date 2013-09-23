# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import jit, float_, double

@jit(float_(float_))
def f(x):
    return x

assert f(None) == None
assert f(2.0) == 2.0

@jit(double(double))
def f(x):
    return x

assert f(None) == None
assert f(2.0) == 2.0