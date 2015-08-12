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


@jit(cache=True, forceobj=True)
def looplifted(n):
    object()
    res = 0
    for i in range(n):
        res = res + i
    return res


Z = 1

# Exercise returning a record instance.  This used to hardcode the dtype
# pointer's value in the bitcode.

packed_record_type = np.dtype([('a', np.int8), ('b', np.float64)])
aligned_record_type = np.dtype([('a', np.int8), ('b', np.float64)], align=True)

packed_arr = np.empty(2, dtype=packed_record_type)
for i in range(packed_arr.size):
    packed_arr[i]['a'] = i + 1
    packed_arr[i]['b'] = i + 42.5

aligned_arr = np.array(packed_arr, dtype=aligned_record_type)

@jit(cache=True, nopython=True)
def record_return(ary, i):
    return ary[i]
