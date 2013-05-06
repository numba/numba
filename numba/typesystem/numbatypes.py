# -*- coding: utf-8 -*-
"""
Shorthands for type constructing, promotions, etc.
"""
from __future__ import print_function, division, absolute_import

from numba.utils import is_builtin
from numba.typesystem import numpy_support, types, universe
from numba.typesystem.types import *

__all__ = ["integral", "floating", "complextypes", "numeric", "native_integral"]

integral = []
floating = []
complextypes = []
numeric = []
native_integral = []

#------------------------------------------------------------------------
# Type shorthands
#------------------------------------------------------------------------

def add_type(name, ty, d=globals()):
    name = name + "_" if is_builtin(name) else name
    d[name] = ty
    assert name not in __all__, name
    __all__.append(name)

# Add some unit types...
for typename in universe.numba_unit_types:
    add_type(typename, mono(typename, typename))

# Add type constructors
for typename, ty in types.numba_type_registry.items():
    add_type(typename, ty)

# Add ints...
add_type("void", mono("void", "void"))
for typename in universe.int_typenames:
    ty = mono("int", typename,
              itemsize=universe.default_type_sizes[typename],
              signed=typename in universe.signed)
    add_type(typename, ty)

    integral.append(ty)
    if universe.is_native_int(typename):
        native_integral.append(ty)

# Add floats...
floats = "float", "double", "longdouble"
aliases = "float32", "float64", "float128"
for typename, alias in zip(floats, aliases):
    ty = mono("float", typename, itemsize=universe.default_type_sizes[typename])
    add_type(typename, ty)
    add_type(alias, ty)
    floating.append(ty)

# Add complexes...
add_type("complex64", complex_(float_))
add_type("complex128", complex_(double))
add_type("complex256", complex_(longdouble))
complextypes.extend([complex64, complex128, complex256])

# ______________________________________________________________________

shortnames = dict(
    O = object_,
    b1 = bool_,
    i1 = int8,
    i2 = int16,
    i4 = int32,
    i8 = int64,
    u1 = uint8,
    u2 = uint16,
    u4 = uint32,
    u8 = uint64,

    f4 = float32,
    f8 = float64,
    f16 = float128,

    c8 = complex64,
    c16 = complex128,
    c32 = complex256,
)

for shortname, ty in shortnames.iteritems():
    add_type(shortname, ty)

# ______________________________________________________________________

numeric.extend(integral + floating + complextypes)
for ty in integral:
    if ty.typename in universe.native_sizes:
        native_integral.append(ty)

#------------------------------------------------------------------------
# Public Type Constructors
#------------------------------------------------------------------------

def from_numpy_dtype(np_dtype):
    """
    :param np_dtype: the NumPy dtype (e.g. np.dtype(np.double))
    :return: a dtype type representation
    """
    return dtype(numpy_support.map_dtype(np_dtype))

def array(dtype, ndim):
    """
    :param dtype: the Numba dtype type (e.g. double)
    :param ndim: the array dimensionality (int)
    :return: an array type representation
    """
    if ndim == 0:
        return dtype
    return ArrayType(dtype, ndim)

def struct_(fields=(), name=None, readonly=False, packed=False, **kwargs):
    "Create a mutable struct type"
    if fields and kwargs:
        raise TypeError("The struct must be either ordered or unordered")
    elif kwargs:
        # fields = sort_types(kwargs)
        fields = list(kwargs.iteritems())

    return MutableStructType(fields, name, readonly, packed)
