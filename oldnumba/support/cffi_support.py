# -*- coding: utf-8 -*-
"""
Support for CFFI. Allows checking whether objects are CFFI functions and
obtaining the pointer and numba signature.
"""
from __future__ import print_function, division, absolute_import

import numba
from numba.typesystem import *

try:
    import cffi
    ffi = cffi.FFI()
except ImportError:
    ffi = None

def is_cffi_func(obj):
    "Check whether the obj is a CFFI function"
    try:
        # return type(obj) is cffi_func_type
        # This is dangerous:
        #   >>> ffi.typeof("void (*)()")
        #   <ctype 'void(*)()'>
        return ffi.typeof(obj).kind == 'function'
    except TypeError:
        return False

def get_pointer(cffi_func):
    """
    Get a pointer to the underlying function for a CFFI function as an
    integer.
    """
    return int(ffi.cast("uintptr_t", cffi_func))

def map_type(cffi_type):
    "Map CFFI type to numba type"
    kind = getattr(cffi_type, 'kind', '')
    if kind in ('struct', 'union'):
        if kind == 'union':
            result = None
        else:
            result = numba.struct([(name, map_type(field.type))
                               for name, field in cffi_type.fields])
    elif kind == 'function':
        restype = map_type(cffi_type.result)
        argtypes = [map_type(arg) for arg in cffi_type.args]
        result = numba.function(restype, argtypes,
                                is_vararg=cffi_type.ellipsis).pointer()
    else:
        result = type_map.get(cffi_type)

    if result is None:
        raise TypeError(cffi_type)

    return result

def get_signature(cffi_func):
    "Get the numba signature for a CFFI function"
    return map_type(ffi.typeof(cffi_func)).base_type

if ffi is None:
    # Disable cffi support
    is_cffi_func = lambda x: False
    type_map = None
else:
    type_map = {
        ffi.typeof('char') :                char,
        ffi.typeof('short') :               short,
        ffi.typeof('int') :                 int_,
        ffi.typeof('long') :                long_,
        ffi.typeof('long long') :           longlong,
        ffi.typeof('unsigned char') :       uchar,
        ffi.typeof('unsigned short') :      ushort,
        ffi.typeof('unsigned int') :        uint,
        ffi.typeof('unsigned long') :       ulong,
        ffi.typeof('unsigned long long') :  ulonglong,
        ffi.typeof('int8_t') :              char,
        ffi.typeof('uint8_t') :             uchar,
        ffi.typeof('int16_t') :             short,
        ffi.typeof('uint16_t') :            ushort,
        ffi.typeof('int32_t') :             int_,
        ffi.typeof('uint32_t') :            uint,
        ffi.typeof('int64_t') :             longlong,
        ffi.typeof('uint64_t') :            ulonglong,
        ffi.typeof('float') :               float_,
        ffi.typeof('double') :              double,
        # ffi.typeof('long double') :         longdouble,
        ffi.typeof('char *') :              c_string_type,
        ffi.typeof('ssize_t') :             Py_ssize_t,
        ffi.typeof('size_t') :              size_t,
    }

    ffi.cdef("int printf(char *, ...);")
    lib = ffi.dlopen(None)
    cffi_func_type = type(lib.printf)
