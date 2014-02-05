# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import cffi_support, jit, double
from cffi import FFI
from math import pi


ffi = FFI()
ffi.cdef('double sin(double x);')

# loads the entire C namespace
C = ffi.dlopen(None)                     
c_sin = C.sin

@jit(nopython=True)
def cffi_sin_example(x):
    return c_sin(x)

print(cffi_sin_example(pi))
