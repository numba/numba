import os
import ctypes
import doctest

from numba import *
import numba

try:
    import cffi
    ffi = cffi.FFI()
except ImportError:
    ffi = None

# ______________________________________________________________________

def test():
    if ffi is not None:
        test_cffi_calls()

# ______________________________________________________________________
# Tests

@autojit(nopython=True)
def call_cffi_func(func, value):
    return func(value)

def test_cffi_calls():
    # Test printf for nopython and no segfault
    ffi.cdef("int printf(char *, ...);", override=True)
    lib = ffi.dlopen(None)
    printf = lib.printf
    call_cffi_func(printf, "Hello world!\n")

# ______________________________________________________________________

if __name__ == "__main__":
    test()