# -*- coding: utf-8 -*-
"""
Shorthands for type constructing, promotions, etc.
"""
from __future__ import print_function, division, absolute_import

import ctypes

from numba.typesystem import types, universe, itypesystem
from numba.typesystem.types import *

__all__ = ["integral", "unsigned_integral", "floating", "complextypes",
           "numeric", "native_integral", "from_numpy_dtype",
           "float_", "double", "longdouble",    # aliases
           "complex64", "complex128", "complex256"]

integral = []
unsigned_integral = []
floating = []
complextypes = []
numeric = []
native_integral = []

#------------------------------------------------------------------------
# All unit types
#------------------------------------------------------------------------

# Add type constructors
for name, ty in types.numba_type_registry.items():
    name = itypesystem.tyname(ty.typename)
    globals()[name] = ty
    __all__.append(name)


def mono(*args, **kwargs):
    ty = types.mono(*args, **kwargs)

    if ty.is_int:
        ty.signed = ty.typename in universe.signed
    if ty.is_int or ty.is_float:
        ty.itemsize = universe.default_type_sizes[ty.typename]

    # Add types to categories numeric, integral, floating, etc...
    if ty.is_int:
        integral.append(ty)
        if not ty.signed:
            unsigned_integral.append(ty)
        if universe.is_native_int(ty.typename):
            native_integral.append(ty)
    elif ty.is_float:
        floating.append(ty)

    if ty.is_numeric:
        numeric.append(ty)

    __all__.append(itypesystem.tyname(ty.typename))
    return ty

# Numeric types
char         = mono("int", "char",         flags=["numeric"])
uchar        = mono("int", "uchar",        flags=["numeric"])
short        = mono("int", "short",        flags=["numeric"])
ushort       = mono("int", "ushort",       flags=["numeric"])
int_         = mono("int", "int",          flags=["numeric"])
uint         = mono("int", "uint",         flags=["numeric"])
long_        = mono("int", "long",         flags=["numeric"])
ulong        = mono("int", "ulong",        flags=["numeric"])
longlong     = mono("int", "longlong",     flags=["numeric"])
ulonglong    = mono("int", "ulonglong",    flags=["numeric"])
int8         = mono("int", "int8",         flags=["numeric"])
int16        = mono("int", "int16",        flags=["numeric"])
int32        = mono("int", "int32",        flags=["numeric"])
int64        = mono("int", "int64",        flags=["numeric"])
uint8        = mono("int", "uint8",        flags=["numeric"])
uint16       = mono("int", "uint16",       flags=["numeric"])
uint32       = mono("int", "uint32",       flags=["numeric"])
uint64       = mono("int", "uint64",       flags=["numeric"])
size_t       = mono("int", "size_t",       flags=["numeric"])
npy_intp     = mono("int", "npy_intp",     flags=["numeric"])
Py_ssize_t   = mono("int", "Py_ssize_t",   flags=["numeric"])
Py_uintptr_t = mono("int", "Py_uintptr_t", flags=["numeric"])

float32      = mono("float", "float32",    flags=["numeric"])
float64      = mono("float", "float64",    flags=["numeric"])
float128     = mono("float", "float128",   flags=["numeric"])
float_, double, longdouble = float32, float64, float128

complex64    = complex_(float32)
complex128   = complex_(float64)
complex256   = complex_(float128)

bool_        = mono("bool", "bool", flags=["int", "numeric"])
null         = mono("null", "null", flags=["pointer"])
void         = mono("void", "void")

obj_type = lambda name: mono(name, name, flags=["object"])

# Add some unit types... (objects)
object_      = obj_type("object")
unicode_     = obj_type("unicode")
none         = obj_type("none")
ellipsis     = obj_type("ellipsis")
slice_       = obj_type("slice")
newaxis      = obj_type("newaxis")
range_       = obj_type("range")
string_      = mono("string", "string", flags=["object", "c_string"])
c_string_type = string_

complextypes.extend([complex64, complex128, complex256])

# ______________________________________________________________________

O = object_
b1 = bool_
i1 = int8
i2 = int16
i4 = int32
i8 = int64
u1 = uint8
u2 = uint16
u4 = uint32
u8 = uint64

f4 = float32
f8 = float64
f16 = float128

c8 = complex64
c16 = complex128
c32 = complex256

#------------------------------------------------------------------------
# Polytype constructors
#------------------------------------------------------------------------

def from_numpy_dtype(np_dtype):
    """
    :param np_dtype: the NumPy dtype (e.g. np.dtype(np.double))
    :return: a dtype type representation
    """
    from numba.typesystem import numpy_support
    return numpy_dtype(numpy_support.map_dtype(np_dtype))

def array(dtype, ndim, is_c_contig=False, is_f_contig=False, inner_contig=False):
    """
    :param dtype: the Numba dtype type (e.g. double)
    :param ndim: the array dimensionality (int)
    :return: an array type representation
    """
    if ndim == 0:
        return dtype
    return ArrayType(dtype, ndim, is_c_contig, is_f_contig, inner_contig)

sort_key = lambda (n, ty): ctypes.sizeof(ty.to_ctypes())

def struct_(fields=(), name=None, readonly=False, packed=False, **kwargs):
    "Create a mutable struct type"
    if fields and kwargs:
        raise TypeError("The struct must be either ordered or unordered")
    elif kwargs:
        import ctypes
        fields = sorted(kwargs.iteritems(), key=sort_key, reverse=True)
        # fields = sort_types(kwargs)
        # fields = list(kwargs.iteritems())

    return MutableStructType(fields, name, readonly, packed)
