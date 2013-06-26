import os
import ctypes

from numba import *

@autojit(nopython=True)
def call_ctypes_func(func, value):
    return func(value)


def test_ctypes_calls():
    # Test puts for no segfault
    libc = ctypes.CDLL(ctypes.util.find_library('c'))
    puts = libc.puts
    puts.argtypes = [ctypes.c_char_p]
    call_ctypes_func(puts, "Hello World!")

    # Test ceil result
    libm = ctypes.CDLL(ctypes.util.find_library('m'))
    ceil = libm.ceil
    ceil.argtypes = [ctypes.c_double]
    ceil.restype = ctypes.c_double
    result = call_ctypes_func(ceil, 10.1)
    assert result == 11.0, result

def test_str_return():
    try:
        import errno
    except ImportError:
        return

    libc = ctypes.CDLL(ctypes.util.find_library('c'))

    strerror = libc.strerror
    strerror.argtypes = [ctypes.c_int]
    strerror.restype = ctypes.c_char_p

    expected = os.strerror(errno.EACCES)
    got = call_ctypes_func(strerror, errno.EACCES)
    assert expected == got, (expected, got)

if __name__ == "__main__":
    test_ctypes_calls()
    test_str_return()
