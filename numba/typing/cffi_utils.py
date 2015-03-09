# -*- coding: utf-8 -*-
"""
Support for CFFI. Allows checking whether objects are CFFI functions and
obtaining the pointer and numba signature.
"""
from __future__ import print_function, division, absolute_import

import ctypes

from numba import types
from . import templates

try:
    import cffi
    ffi = cffi.FFI()
except ImportError:
    ffi = None

SUPPORTED = ffi is not None


def is_cffi_func(obj):
    """Check whether the obj is a CFFI function"""
    try:
        return ffi.typeof(obj).kind == 'function'
    except TypeError:
        return False

def get_pointer(cffi_func):
    """
    Get a pointer to the underlying function for a CFFI function as an
    integer.
    """
    return int(ffi.cast("uintptr_t", cffi_func))


_cached_type_map = None

def _type_map():
    """
    Lazily compute type map, as calling ffi.typeof() involves costly
    parsing of C code...
    """
    global _cached_type_map
    if _cached_type_map is None:
        _cached_type_map = {
            ffi.typeof('char') :                types.int8,
            ffi.typeof('short') :               types.short,
            ffi.typeof('int') :                 types.intc,
            ffi.typeof('long') :                types.long_,
            ffi.typeof('long long') :           types.longlong,
            ffi.typeof('unsigned char') :       types.uchar,
            ffi.typeof('unsigned short') :      types.ushort,
            ffi.typeof('unsigned int') :        types.uintc,
            ffi.typeof('unsigned long') :       types.ulong,
            ffi.typeof('unsigned long long') :  types.ulonglong,
            ffi.typeof('int8_t') :              types.char,
            ffi.typeof('uint8_t') :             types.uchar,
            ffi.typeof('int16_t') :             types.short,
            ffi.typeof('uint16_t') :            types.ushort,
            ffi.typeof('int32_t') :             types.intc,
            ffi.typeof('uint32_t') :            types.uintc,
            ffi.typeof('int64_t') :             types.longlong,
            ffi.typeof('uint64_t') :            types.ulonglong,
            ffi.typeof('float') :               types.float_,
            ffi.typeof('double') :              types.double,
            # ffi.typeof('long double') :         longdouble,
            ffi.typeof('char *') :              types.voidptr,
            ffi.typeof('void *') :              types.voidptr,
            ffi.typeof('uint8_t *') :           types.CPointer(types.uint8),
            ffi.typeof('ssize_t') :             types.intp,
            ffi.typeof('size_t') :              types.uintp,
            ffi.typeof('void') :                types.void,
        }
    return _cached_type_map


def map_type(cffi_type):
    """
    Map CFFI type to numba type.
    """
    kind = getattr(cffi_type, 'kind', '')
    if kind in ('struct', 'union'):
        raise TypeError("No support for struct or union")
    elif kind == 'function':
        if cffi_type.ellipsis:
            raise TypeError("vararg function is not supported")
        restype = map_type(cffi_type.result)
        argtypes = [map_type(arg) for arg in cffi_type.args]
        return templates.signature(restype, *argtypes)
    else:
        result = _type_map().get(cffi_type)

    if result is None:
        raise TypeError(cffi_type)

    return result


def make_function_type(cffi_func):
    """
    Return a Numba type for the given CFFI function pointer.
    """
    cffi_type = ffi.typeof(cffi_func)
    sig = map_type(cffi_type)
    return types.ExternalFunctionPointer(sig, get_pointer=get_pointer)


class ExternCFunction(types.ExternalFunction):
    # XXX unused?

    def __init__(self, symbol, cstring):
        """Parse C function declaration/signature"""
        parser = cffi.cparser.Parser()
        rft = parser.parse_type(cstring) # "RawFunctionType"
        type_map = _type_map()
        self.restype = type_map[rft.result.build_backend_type(ffi, None)]
        self.argtypes = [type_map[arg.build_backend_type(ffi, None)] for arg in rft.args]
        signature = templates.signature(self.restype, *self.argtypes)
        super(ExternCFunction, self).__init__(symbol, signature)
