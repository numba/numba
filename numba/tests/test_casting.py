import sys
import os
import numpy as np
import ctypes

from numba import *
import numba

@autojit(backend='ast')
def cast_int():
    value = 1.7
    return int32(value)

@autojit(backend='ast')
def cast_complex():
    value = 1.2
    return complex128(value)

@autojit(backend='ast')
def cast_float():
    value = 5
    return float_(value)

@autojit(backend='ast')
def cast_object(dst_type):
    value = np.arange(10, dtype=np.double)
    return dst_type(value)

@autojit(backend='ast')
def cast_as_numba_type_attribute():
    value = 4.4
    return numba.int32(value)

def cast_in_python():
    return int_(10) == 10

def test_casts():
    assert cast_int() == 1
    assert cast_complex() == 1.2 + 0j
    assert cast_float() == 5.0
    value = cast_object(double[:])
    # print sys.getrefcount(value), value, np.arange(10, dtype=np.double)
    assert np.all(value == np.arange(10, dtype=np.double)), value
    assert cast_as_numba_type_attribute() == 4
    assert cast_in_python()

if __name__ == "__main__":
    test_casts()
