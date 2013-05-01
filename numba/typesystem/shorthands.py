# -*- coding: utf-8 -*-
"""
Shorthands for type constructing, promotions, etc.
"""
from __future__ import print_function, division, absolute_import

from itertools import chain

from numba.utils import is_builtin
from numba.typesystem import numpy_support, types, universe
from numba.typesystem.defaults import numba_universe as u

__all__ = []

#------------------------------------------------------------------------
# Type shorthands
#------------------------------------------------------------------------

# Set numba universe types as globals
d = globals()
for name in chain(u.monotypes, u.polytypes, ["struct"], universe.complex_typenames):
    ty = getattr(u, name)
    name = name + "_" if is_builtin(name) else name
    __all__.append(name)
    d[name] = ty

# print(struct([('a', int_)]))
# print(function(int_, (float_, double, complex128)))

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
# Type Constructor Shorthands
#------------------------------------------------------------------------

def from_numpy_dtype(np_dtype):
    """
    :param np_dtype: the NumPy dtype (e.g. np.dtype(np.double))
    :return: a dtype type representation
    """
    return dtype(numpy_support.map_dtype(np_dtype))

def dtype(dtype_type):
    """

    :param dtype: the Numba dtype type (e.g. double)
    :return: a dtype type representation
    """
    assert isinstance(dtype_type, types.Type)
    return u.dtype(dtype_type)

def array(dtype, ndim):
    """
    :param dtype: the Numba dtype type (e.g. double)
    :param ndim: the array dimensionality (int)
    :return: an array type representation
    """
    if ndim == 0:
        return dtype
    return u.array(dtype, ndim)

def tuple_(base_type, size=-1):
    """
    :param base_type: the element type of the tuple
    :param size: set to a value >= 0 is the size is known
    :return: a tuple type representation
    """
    return u.tuple(base_type, size)

def list_(base_type, size=-1):
    """
    :param base_type: the element type of the tuple
    :param size: set to a value >= 0 is the size is known
    :return: a tuple type representation
    """
    return u.list(base_type, size)
