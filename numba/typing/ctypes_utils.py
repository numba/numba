"""
Support for typing ctypes function pointers.
"""

from __future__ import absolute_import

import ctypes
import sys

from numba import types
from . import templates


CTYPES_MAP = {
    None: types.none,
    ctypes.c_bool: types.boolean,
    
    ctypes.c_int8:  types.int8,
    ctypes.c_int16: types.int16,
    ctypes.c_int32: types.int32,
    ctypes.c_int64: types.int64,

    ctypes.c_uint8: types.uint8,
    ctypes.c_uint16: types.uint16,
    ctypes.c_uint32: types.uint32,
    ctypes.c_uint64: types.uint64,

    ctypes.c_float: types.float32,
    ctypes.c_double: types.float64,

    ctypes.c_void_p: types.voidptr,
    ctypes.py_object: types.ffi_forced_object,
}


def convert_ctypes(ctypeobj):
    try:
        return CTYPES_MAP[ctypeobj]
    except KeyError:
        raise TypeError("unhandled ctypes type: %s" % ctypeobj)


def is_ctypes_funcptr(obj):
    try:
        # Is it something of which we can get the address
        ctypes.cast(obj, ctypes.c_void_p)
    except ctypes.ArgumentError:
        return False
    else:
        # Does it define argtypes and restype
        return hasattr(obj, 'argtypes') and hasattr(obj, 'restype')


def get_pointer(ctypes_func):
    """
    Get a pointer to the underlying function for a ctypes function as an
    integer.
    """
    return ctypes.cast(ctypes_func, ctypes.c_void_p).value


def make_function_type(cfnptr):
    """
    Return a Numba type for the given ctypes function pointer.
    """
    if cfnptr.argtypes is None:
        raise TypeError("ctypes function %r doesn't define its argument types; "
                        "consider setting the `argtypes` attribute"
                        % (cfnptr.__name__,))
    cargs = [convert_ctypes(a)
             for a in cfnptr.argtypes]
    cret = convert_ctypes(cfnptr.restype)
    if sys.platform == 'win32' and not cfnptr._flags_ & ctypes._FUNCFLAG_CDECL:
        # 'stdcall' calling convention under Windows
        cconv = 'x86_stdcallcc'
    else:
        # Default C calling convention
        cconv = None

    sig = templates.signature(cret, *cargs)
    return types.ExternalFunctionPointer(sig, cconv=cconv,
                                         get_pointer=get_pointer)
