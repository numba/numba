"""
This file will be copied to a temporary directory in order to
exercise caching compiled Numba functions.

See test_dispatcher.py.
"""


import numpy as np

from numba import jit

from numba.tests.ctypes_usecases import c_sin


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


@jit(cache=True, nopython=True)
def use_c_sin(x):
    return c_sin(x)


@jit(cache=True, nopython=True)
def ambiguous_function(x):
    return x + 2

renamed_function1 = ambiguous_function

@jit(cache=True, nopython=True)
def ambiguous_function(x):
    return x + 6

renamed_function2 = ambiguous_function


def make_closure(x):
    @jit(cache=True, nopython=True)
    def closure(y):
        return x + y

    return closure

closure1 = make_closure(3)
closure2 = make_closure(5)


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
