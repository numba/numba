# -*- coding: utf-8 -*-
"""
Shorthands for type constructing, promotions, etc.
"""
from __future__ import print_function, division, absolute_import
import __future__

import inspect

from numba.typesystem import types, universe
from numba.typesystem.types import *

__all__ = [] # set below

integral = []
unsigned_integral = []
floating = []
complextypes = []
numeric = []
native_integral = []
datetimetypes = []

domain_name = "numba"

ranking = ["bool", "int", "float", "complex", "object"]

def rank(type):
    return ranking.index(type.kind)

#------------------------------------------------------------------------
# All unit types
#------------------------------------------------------------------------

def unit(*args, **kwargs):
    ty = types.unit(*args, **kwargs)

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

    return ty

# Numeric types
char         = unit("int", "char",         flags=["numeric"])
uchar        = unit("int", "uchar",        flags=["numeric"])
short        = unit("int", "short",        flags=["numeric"])
ushort       = unit("int", "ushort",       flags=["numeric"])
int_         = unit("int", "int",          flags=["numeric"])
uint         = unit("int", "uint",         flags=["numeric"])
long_        = unit("int", "long",         flags=["numeric"])
ulong        = unit("int", "ulong",        flags=["numeric"])
longlong     = unit("int", "longlong",     flags=["numeric"])
ulonglong    = unit("int", "ulonglong",    flags=["numeric"])
int8         = unit("int", "int8",         flags=["numeric"])
int16        = unit("int", "int16",        flags=["numeric"])
int32        = unit("int", "int32",        flags=["numeric"])
int64        = unit("int", "int64",        flags=["numeric"])
uint8        = unit("int", "uint8",        flags=["numeric"])
uint16       = unit("int", "uint16",       flags=["numeric"])
uint32       = unit("int", "uint32",       flags=["numeric"])
uint64       = unit("int", "uint64",       flags=["numeric"])
size_t       = unit("int", "size_t",       flags=["numeric"])
npy_intp     = unit("int", "npy_intp",     flags=["numeric"])
Py_ssize_t   = unit("int", "Py_ssize_t",   flags=["numeric"])
Py_uintptr_t = unit("int", "Py_uintptr_t", flags=["numeric"])

float32      = unit("float", "float32",    flags=["numeric"])
float64      = unit("float", "float64",    flags=["numeric"])
# float128     = unit("float", "float128",   flags=["numeric"])
# float_, double, longdouble = float32, float64, float128
float_, double = float32, float64

complex64    = complex_(float32)
complex128   = complex_(float64)
# complex256   = complex_(float128)

bool_        = unit("bool", "bool", flags=["int", "numeric"])
null         = unit("null", "null", flags=["pointer"])
void         = unit("void", "void")

obj_type = lambda name: unit(name, name, flags=["object"])

# Add some unit types... (objects)
object_      = obj_type("object")
unicode_     = obj_type("unicode")
none         = obj_type("none")
ellipsis     = obj_type("ellipsis")
slice_       = obj_type("slice")
newaxis      = obj_type("newaxis")
range_       = obj_type("range")
string_      = unit("string", "string", flags=[#"object",
                                               "c_string"])
c_string_type = string_

complextypes.extend([complex64, complex128]) #, complex256])

tuple_of_obj       = tuple_(object_, -1)
list_of_obj        = list_(object_, -1)
dict_of_obj        = dict_(object_, object_, -1)


def datetime(units=None, numpy=True):
    if units not in ['Y', 'M', 'D', 'h', 'm', 's']:
        units = None 
    datetime_type = datetime_(int64, int32, units)
    datetime_type.is_numpy_datetime = numpy
    return datetime_type

def timedelta(units=None, numpy=True):
    if units not in ['Y', 'M', 'D', 'h', 'm', 's']:
        units = None 
    timedelta_type = timedelta_(int64, int32, units)
    timedelta_type.is_numpy_timedelta = numpy
    return timedelta_type

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
# f16 = float128

c8 = complex64
c16 = complex128
# c32 = complex256

for name, value in list(globals().iteritems()): # TODO: Do this better
    if (not isinstance(value, __future__ ._Feature) and not
        inspect.ismodule(value) and not name.startswith("_")):
        __all__.append(name)