# -*- coding: utf-8 -*-
"""
Support for CFFI. Allows checking whether objects are CFFI functions and
obtaining the pointer and numba signature.
"""
from __future__ import print_function, division, absolute_import

from types import BuiltinFunctionType
import ctypes

from numba import types
from . import templates

try:
    import cffi
    ffi = cffi.FFI()
except ImportError:
    ffi = None

SUPPORTED = ffi is not None
_ool_func_types = {}
_ool_func_ptr = {}
_ffi_instances = set()


def is_ffi_instance(obj):
    # Compiled FFI modules have a member, ffi, which is an instance of
    # CompiledFFI, which behaves similarly to an instance of cffi.FFI. In
    # order to simplify handling a CompiledFFI object, we treat them as
    # if they're cffi.FFI instances for typing and lowering purposes.
    try:
        return obj in _ffi_instances or isinstance(obj, cffi.FFI)
    except TypeError: # Unhashable type possible
        return False

def is_cffi_func(obj):
    """Check whether the obj is a CFFI function"""
    try:
        return ffi.typeof(obj).kind == 'function'
    except TypeError:
        try:
            return obj in _ool_func_types
        except:
            return False

def get_pointer(cffi_func):
    """
    Get a pointer to the underlying function for a CFFI function as an
    integer.
    """
    if cffi_func in _ool_func_ptr:
        return _ool_func_ptr[cffi_func]
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
            ffi.typeof('char *') :              types.voidptr,
            ffi.typeof('void *') :              types.voidptr,
            ffi.typeof('uint8_t *') :           types.CPointer(types.uint8),
            ffi.typeof('float *') :             types.CPointer(types.float32),
            ffi.typeof('double *') :            types.CPointer(types.float64),
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
    cffi_type = _ool_func_types.get(cffi_func) or ffi.typeof(cffi_func)
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


registry = templates.Registry()

@registry.register
class FFI_from_buffer(templates.AbstractTemplate):
    key = 'ffi.from_buffer'

    def generic(self, args, kws):
        if kws or (len(args) != 1):
            return
        [ary] = args
        if not (isinstance(ary, types.Array) and ary.layout in ('C', 'F')):
            return
        ptr = types.CPointer(ary.dtype)
        return templates.signature(ptr, ary)

@registry.register_attr
class FFIAttribute(templates.AttributeTemplate):
    key = types.ffi

    def resolve_from_buffer(self, ffi):
        return types.BoundFunction(FFI_from_buffer, types.ffi)


def register_module(mod):
    """
    Add typing for all functions in an out-of-line CFFI module to the typemap
    """
    for f in dir(mod.lib):
        f = getattr(mod.lib, f)
        if isinstance(f, BuiltinFunctionType):
            _ool_func_types[f] = mod.ffi.typeof(f)
            addr = mod.ffi.addressof(mod.lib, f.__name__)
            _ool_func_ptr[f] = int(mod.ffi.cast("uintptr_t", addr))
        _ffi_instances.add(mod.ffi)
