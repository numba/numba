"""
Infer types for NumPy functionality. This includes:

    1) Figuring out dtypes

        e.g. np.double     -> double
             np.dtype('d') -> double

    2) Function calls such as np.empty/np.empty_like/np.arange/etc
"""

import numpy as np

from numba import *
from numba.minivect import minitypes
from numba import typesystem
from numba.type_inference.module_type_inference import register, register_inferer
from numba.typesystem import get_type


#------------------------------------------------------------------------
# Type Definitions
#------------------------------------------------------------------------

index_array_t = npy_intp[:]

#------------------------------------------------------------------------
# Some utilities
#------------------------------------------------------------------------

def resolve_attribute_dtype(dtype, default=None):
    "Resolve the type for numpy dtype attributes"
    if dtype.is_numpy_dtype:
        return dtype

    if dtype.is_numpy_attribute:
        numpy_attr = getattr(dtype.module, dtype.attr, None)
        if isinstance(numpy_attr, np.dtype):
            return typesystem.from_numpy_dtype(numpy_attr)
        elif issubclass(numpy_attr, np.generic):
            return typesystem.from_numpy_dtype(np.dtype(numpy_attr))

def get_dtype(dtype_arg, default_dtype=None):
    """
    Simple helper function to map an AST node dtype keyword
    argument => NumPy dtype.
    '"""
    if dtype_arg is None:
        if default_dtype is None:
            return None

        return typesystem.dtype(default_dtype)
    else:
        return resolve_attribute_dtype(dtype_arg)

def promote_to_array(dtype):
    "Promote scalar to 0d array type"
    if not dtype.is_array:
        dtype = minitypes.ArrayType(dtype, 0)
    return dtype

def array_from_object(a):
    """
    object -> array type:

        array_from_object(ASTNode([[1, 2], [3, 4]])) => int64[:, :]
    """
    return array_from_type(a)

def array_from_type(type):
    if type.is_array:
        return type
    elif type.is_tuple or type.is_list:
        dtype = array_from_object(type.dtype)
        if dtype.is_array:
            type = dtype.copy()
            type.ndim += 1
            return type
    elif not type.is_object:
        return minitypes.ArrayType(dtype=type, ndim=0)

    return object_

#------------------------------------------------------------------------
# Resolution of NumPy calls
#------------------------------------------------------------------------

@register(np)
def dtype(obj, align):
    "Parse np.dtype(...) calls"
    if obj is None:
        return None

    return get_dtype(obj)

def empty_like(a, dtype, order):
    "Parse the result type for np.empty_like calls"
    if a is None:
        return None

    if a.is_array:
        if dtype:
            dtype_type = get_dtype(dtype)
            if dtype_type is None:
                return a
            dtype = dtype_type.dtype
        else:
            dtype = a.dtype

        return typesystem.array(dtype, a.ndim)

register_inferer(np, 'empty_like', empty_like)
register_inferer(np, 'zeros_like', empty_like)
register_inferer(np, 'ones_like', empty_like)

def empty(shape, dtype, order):
    if shape is None:
        return None

    dtype = get_dtype(dtype, float64)

    if shape.is_int:
        ndim = 1
    elif shape.is_tuple or shape.is_list:
        ndim = shape.size
    else:
        return None

    return typesystem.array(dtype.dtype, ndim)

register_inferer(np, 'empty', empty)
register_inferer(np, 'zeros', empty)
register_inferer(np, 'ones', empty)

@register(np)
def arange(start, stop, step, dtype):
    "Resolve a call to np.arange()"
    # NOTE: dtype must be passed as a keyword argument, or as the fourth
    # parameter
    dtype_type = get_dtype(dtype, int64)
    if dtype_type is not None:
        # return a 1D array type of the given dtype
        return dtype_type.dtype[:]

@register(np)
def dot(context, a, b, out):
    "Resolve a call to np.dot()"
    if out is not None:
        return out

    lhs_type = promote_to_array(a)
    rhs_type = promote_to_array(b)

    dtype = context.promote_types(lhs_type.dtype, rhs_type.dtype)
    dst_ndim = lhs_type.ndim + rhs_type.ndim - 2

    result_type = typesystem.array(dtype, dst_ndim)
    return result_type

@register(np)
def array(object, dtype, order, subok):
    type = array_from_object(object)
    if dtype is not None:
        type = type.copy(dtype=dtype)

    return type

@register(np)
def nonzero(a):
    return _nonzero(array_from_object(a))

def _nonzero(type):
    if type.is_array:
        return typesystem.tuple_(index_array_t, type.ndim)
    else:
        return typesystem.tuple_(index_array_t)

@register(np)
def where(context, condition, x, y):
    if x is None and y is None:
        return nonzero(condition)

    xtype = array_from_object(x)
    ytype = array_from_object(y)
    type = context.promote_types(xtype, ytype)
    return type

def reduce_(a, axis, dtype, out):
    if out is not None:
        return out

    dtype_type = get_dtype(dtype, default_dtype=a.dtype).dtype

    if axis is None:
        # Return the scalar type
        return dtype_type

    # Handle the axis parameter
    if axis.is_tuple and axis.is_sized:
        # axis=(tuple with a constant size)
        return typesystem.array(dtype_type, a.ndim - axis.size)
    elif axis.is_int:
        # axis=1
        return typesystem.array(dtype_type, a.ndim - 1)
    else:
        # axis=(something unknown)
        return object_

register_inferer(np, 'sum', reduce_)
register_inferer(np, 'prod', reduce_)
