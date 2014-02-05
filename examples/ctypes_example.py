# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division
from ctypes import *
import sys
from math import pi
from numba import jit, double


is_windows = sys.platform.startswith('win32')
if is_windows:
    raise OSError('Example does not work on Windows platforms yet.')


proc = CDLL(None)

c_sin = proc.sin
c_sin.argtypes = [c_double]
c_sin.restype = c_double

def use_c_sin(x):
    return c_sin(x)


ctype_wrapping = CFUNCTYPE(c_double, c_double)(use_c_sin)

def use_ctype_wrapping(x):
    return ctype_wrapping(x)


cfunc = jit(double(double))(use_c_sin)
print(cfunc(pi))

cfunc = jit(double(double))(use_ctype_wrapping)
print(cfunc(pi))


