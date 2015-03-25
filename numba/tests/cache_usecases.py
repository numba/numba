"""
This file will be copied to a temporary directory in order to
exercise caching compiled Numba functions.

See test_dispatcher.py.
"""


import numpy as np

from numba import jit


@jit(cache=True, nopython=True)
def add_usecase(x, y):
    return x + y + Z


@jit(cache=True, forceobj=True)
def add_objmode_usecase(x, y):
    object()
    return x + y + Z


@jit(nopython=True)
def add_nocache_usecase(x, y):
    return x + y + Z


@jit(cache=True, nopython=True)
def inner(x, y):
    return x + y + Z

@jit(cache=True, nopython=True)
def outer(x, y):
    return inner(-y, x)


Z = 1

