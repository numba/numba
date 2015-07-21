from __future__ import print_function, division, absolute_import

from numba import cffi_support


if cffi_support.SUPPORTED:
    from cffi import FFI
    ffi = FFI()
    ffi.cdef("""
    double sin(double x);
    double cos(double x);
    """)
    C = ffi.dlopen(None)                     # loads the entire C namespace
    cffi_sin = C.sin
    cffi_cos = C.cos


def use_cffi_sin(x):
    return cffi_sin(x) * 2

def use_two_funcs(x):
    return cffi_sin(x) - cffi_cos(x)

def use_func_pointer(fa, fb, x):
    if x > 0:
        return fa(x)
    else:
        return fb(x)
