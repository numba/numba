# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numba
from numba import *
import numpy as np

int32p = int32.pointer()
voidp = void.pointer()

@jit
def test_pointer_arithmetic():
    """
    >>> test_pointer_arithmetic()
    48L
    """
    p = int32p(Py_uintptr_t(0))
    p = p + 10
    p += 2
    return Py_uintptr_t(p) # 0 + 4 * 12

@jit(locals={"pointer_value": Py_uintptr_t})
def test_pointer_indexing(pointer_value, type_p):
    """
    >>> a = np.array([1, 2, 3, 4], dtype=np.float32)
    >>> test_pointer_indexing(a.ctypes.data, float32.pointer())
    (1.0, 2.0, 3.0, 4.0)

    >>> a = np.array([1, 2, 3, 4], dtype=np.int64)
    >>> test_pointer_indexing(a.ctypes.data, int64.pointer())
    (1L, 2L, 3L, 4L)
    """
    p = type_p(pointer_value)
    return p[0], p[1], p[2], p[3]

@jit
def test_compare_null():
    """
    >>> test_compare_null()
    True
    """
    return voidp(Py_uintptr_t(0)) == numba.NULL

test_pointer_arithmetic()
