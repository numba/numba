# -*- coding: utf-8 -*-
"""
Shorthands for type constructing, promotions, etc.
"""
from __future__ import print_function, division, absolute_import

from itertools import chain

from numba.utils import is_builtin
from numba.typesystem import numpy_support, types, universe
from numba.typesystem.defaults import numba_universe as u
from numba.typesystem.types import *

__all__ = []

#------------------------------------------------------------------------
# Public Type Constructors
#------------------------------------------------------------------------

complex_ = ComplexType
tuple_ = TupleType
list_ = ListType
type_ = MetaType

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

#------------------------------------------------------------------------
# Type shorthands
#------------------------------------------------------------------------

# Set numba universe types as globals
d = globals()
for name, ty in u.iter_types():
    name = name + "_" if is_builtin(name) else name
    __all__.append(name)
    d[name] = ty

# ______________________________________________________________________

complex64 = complex_(float_)
complex128 = complex_(double)
complex256 = complex_(longdouble)

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
