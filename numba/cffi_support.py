# -*- coding: utf-8 -*-
"""
Support for CFFI. Allows checking whether objects are CFFI functions and
obtaining the pointer and numba signature.
"""
from __future__ import print_function, division, absolute_import

from numba import types, typing

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


def map_type(cffi_type):
    """Map CFFI type to numba type"""
    kind = getattr(cffi_type, 'kind', '')
    if kind in ('struct', 'union'):
        raise TypeError("No support for struct or union")
    elif kind == 'function':
        if cffi_type.ellipsis:
            raise TypeError("vararg function is not supported")
        restype = map_type(cffi_type.result)
        argtypes = [map_type(arg) for arg in cffi_type.args]
        return typing.signature(restype, *argtypes)
    else:
        result = type_map.get(cffi_type)

    if result is None:
        raise TypeError(cffi_type)

    return result


def make_function_type(cffi_func):
    cffi_type = ffi.typeof(cffi_func)
    signature = map_type(cffi_type)
    cases = [signature]
    template = typing.make_concrete_template("CFFIFuncPtr", cffi_func, cases)
    result = types.FunctionPointer(template, get_pointer(cffi_func))
    return result


if ffi is not None:
    type_map = {
        ffi.typeof('char') :                types.int8,
        ffi.typeof('short') :               types.short,
        ffi.typeof('int') :                 types.int_,
        ffi.typeof('long') :                types.long_,
        ffi.typeof('long long') :           types.longlong,
        ffi.typeof('unsigned char') :       types.uchar,
        ffi.typeof('unsigned short') :      types.ushort,
        ffi.typeof('unsigned int') :        types.uint,
        ffi.typeof('unsigned long') :       types.ulong,
        ffi.typeof('unsigned long long') :  types.ulonglong,
        ffi.typeof('int8_t') :              types.char,
        ffi.typeof('uint8_t') :             types.uchar,
        ffi.typeof('int16_t') :             types.short,
        ffi.typeof('uint16_t') :            types.ushort,
        ffi.typeof('int32_t') :             types.int_,
        ffi.typeof('uint32_t') :            types.uint,
        ffi.typeof('int64_t') :             types.longlong,
        ffi.typeof('uint64_t') :            types.ulonglong,
        ffi.typeof('float') :               types.float_,
        ffi.typeof('double') :              types.double,
        # ffi.typeof('long double') :         longdouble,
        ffi.typeof('char *') :              types.voidptr,
        ffi.typeof('ssize_t') :             types.intp,
        ffi.typeof('size_t') :              types.uintp,
    }

