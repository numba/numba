# -*- coding: utf-8 -*-

"""
Test numba.addressof().
"""

from __future__ import print_function, division, absolute_import

import ctypes

import numba
from numba import *


@jit(int32(int32, int32))
def func(a, b):
    return a * b

@autojit
def error_func():
    pass

# TODO: struct pointer support

before_computed_column = struct_([
    ('x', float32),
    ('y', float32)])

with_computed_column = struct_([
    ('mean', float32),
    ('x', float32),
    ('y', float32)])

signature = void(with_computed_column.ref(),
                 before_computed_column.ref())

# @jit(signature, nopython=True)
def cc_kernel(dst, src):
    dst.mean = (src.x + src.y) / 2.0
    dst.x = src.x
    dst.y = src.y

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

def test_addressof(arg):
    """
    >>> func = test_addressof(func)
    >>> assert func.restype == ctypes.c_int32
    >>> assert func.argtypes == (ctypes.c_int32, ctypes.c_int32)
    >>> func(5, 2)
    10
    """
    return numba.addressof(arg)

def test_addressof_error(arg, **kwds):
    """
    >>> test_addressof_error(error_func)
    Traceback (most recent call last):
        ...
    TypeError: Object is not a jit function

    >>> test_addressof_error(func, propagate=False)
    Traceback (most recent call last):
        ...
    ValueError: Writing unraisable exceptions is not yet supported
    """
    return numba.addressof(arg, **kwds)

def test_address_of_struct_function():
    S1 = before_computed_column.to_ctypes()
    S2 = with_computed_column.to_ctypes()
    ctypes_kernel = numba.addressof(cc_kernel)

    s1 = S1(10, 5)
    s2 = S2(0, 0, 0)
    ctypes_kernel(s2, s1)

    assert s2.x == s1.x
    assert s2.y == s1.y
    assert s2.mean == (s1.x + s1.y) / 2.0


numba.testing.testmod()
