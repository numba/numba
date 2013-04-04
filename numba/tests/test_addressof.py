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
    ValueError: Writing unraisable exception is not yet supported
    """
    return numba.addressof(arg, **kwds)


numba.testmod()
