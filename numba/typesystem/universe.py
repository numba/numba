# -*- coding: utf-8 -*-

"""
Universes of type constructors for numba.
"""

from __future__ import print_function, division, absolute_import

import struct as struct_module
import ctypes

from numba.utils import is_builtin

import numpy as np

def tyname(name):
    return name + "_" if is_builtin(name) else name

names = lambda *names: list(map(tyname, names))

int_typenames = names(
    'char', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong',
    'longlong', 'ulonglong', 'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
    'size_t', 'npy_intp', 'Py_ssize_t', 'Py_uintptr_t',
    'bool', # hmm
)

signed = frozenset(names(
    'char', 'short', 'int', 'long', 'longlong',
    'int8', 'int16', 'int32', 'int64',
    'Py_ssize_t',
))

float_typenames = names(
    'float', 'double', 'longdouble', 'float32', 'float64', 'float128',
)

complex_typenames = names(
    'complex64', 'complex128', 'complex256',
)

numba_unit_types = names(
    "object", "null", "none", "ellipsis", "slice", "newaxis", "range",
)

#------------------------------------------------------------------------
# Default type sizes
#------------------------------------------------------------------------

_plat_bits = struct_module.calcsize('@P') * 8

def getsize(ctypes_name, default):
    try:
        return ctypes.sizeof(getattr(ctypes, ctypes_name))
    except ImportError:
        return default

# Type sizes in bytes
type_sizes = {
    "bool_":        1,
    # Int
    "char":         1,
    "int8":         1,
    "int16":        2,
    "int32":        4,
    "int64":        8,
    # Unsigned int
    "uchar":        1,
    "uint8":        1,
    "uint16":       2,
    "uint32":       4,
    "uint64":       8,
    # Float
    # "float16":      2,
    "float32":      4,
    "float64":      8,
    "float128":     16,
    "float_":       4,
    "double":       8,
    # Complex
    "complex64":    8,
    "complex128":   16,
    "complex256":   32,
}

ctypes_npy_intp = np.empty(0).ctypes.strides._type_

native_sizes = {
    # Int
    "short":        struct_module.calcsize("h"),
    "int_":         struct_module.calcsize("i"),
    "long_":        struct_module.calcsize("l"),
    "longlong":     struct_module.calcsize("Q"),
    "Py_ssize_t":   getsize('c_size_t', _plat_bits // 8),
    "npy_intp":     ctypes.sizeof(ctypes_npy_intp),
    # Unsigned int
    "ushort":       struct_module.calcsize("H"),
    "uint":         struct_module.calcsize("I"),
    "ulong":        struct_module.calcsize("L"),
    "ulonglong":    struct_module.calcsize("Q"),
    "size_t":       getsize('c_size_t', _plat_bits // 8),
    "Py_uintptr_t": ctypes.sizeof(ctypes.c_void_p),
    # Float
    "longdouble":   ctypes.sizeof(ctypes.c_longdouble),
    # Pointer
    "pointer":      ctypes.sizeof(ctypes.c_void_p),
}

default_type_sizes = dict(type_sizes, **native_sizes)

is_native_int = native_sizes.__contains__