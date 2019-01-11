print('=' * 80)

from cffi import FFI
from numba import _helperlib


defs = """
void _numba_test_dict();
"""

ffi = FFI()
ffi.cdef(defs)
# Load the _helperlib namespace


lib = ffi.dlopen(_helperlib.__file__)
lib._numba_test_dict()



