******************
Interfacing with C
******************

Numba supports calling C functions through CFFI and ctypes.

====
CFFI
====

Numba supports calling C functions wrapped with CFFI::

    from numba import jit
    from cffi import FFI

    ffi = FFI()
    ffi.cdef('double sin(double x);')

    # loads the entire C namespace
    C = ffi.dlopen(None)                     
    c_sin = C.sin

    @jit(nopython=True)
    def cffi_sin_example(x):
        return c_sin(x)

======
ctypes
======

Numba also supports calling C functions wrapped with ctypes::

    # This example doesn't work on Windows platforms
    from ctypes import *
    from math import pi
    from numba import jit, double

    proc = CDLL(None)

    c_sin = proc.sin
    c_sin.argtypes = [c_double]
    c_sin.restype = c_double

    @jit
    def use_c_sin(x):
        return c_sin(x)

    ctype_wrapping = CFUNCTYPE(c_double, c_double)(use_c_sin)

    @jit
    def use_ctype_wrapping(x):
        return ctype_wrapping(x)

