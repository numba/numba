import os
import ctypes

from numba import *
import numba

try:
    import cffi
    ffi = cffi.FFI()
except ImportError:
    ffi = None

root = os.path.dirname(os.path.abspath(__file__))
include_dirs = [root]

# ______________________________________________________________________

def test():
    if ffi is not None:
        run_callback()

# ______________________________________________________________________
# Tests

@jit(int_(int_, int_))
def numba_callback(a, b):
    return a * b

@autojit #(nopython=True)
def call_cffi_func(eat_callback):
    return eat_callback(numba_callback)

def run_callback():
    # Define C function taking a callback
    ffi.cdef("typedef int (*callback_t)(int, int);")
    ffi.cdef("typedef int (*eat_callback_t)(callback_t);")
    ffi.cdef("eat_callback_t get_eat_callback();", override=True)

    ffi.cdef("int printf(char *, ...);")
    lib = ffi.verify("""
        typedef int (*callback_t)(int, int);
        typedef int (*eat_callback_t)(callback_t);

        int eat_callback(callback_t callback) {
            return callback(5, 2);
        }

        eat_callback_t get_eat_callback() {
            return (callback_t) eat_callback;
        }
    """, include_dirs=include_dirs)

    # CFFI returns builtin methods instead of CData functions for
    # non-external functions. Get the CDATA function through an indirection.
    eat_callback = lib.get_eat_callback()

    assert call_cffi_func(eat_callback) == 10


# ______________________________________________________________________

if __name__ == "__main__":
    test()
