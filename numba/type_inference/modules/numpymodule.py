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


#------------------------------------------------------------------------
# Type Definitions
#------------------------------------------------------------------------

index_array_t = npy_intp[::1]

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
            return typesystem.NumpyDtypeType(dtype=numpy_attr)
        elif issubclass(numpy_attr, np.generic):
            return typesystem.NumpyDtypeType(dtype=np.dtype(numpy_attr))

def get_dtype(dtype_arg, default_dtype=None):
    "Get the dtype keyword argument from a call to a numpy attribute."
    if dtype_arg is None:
        if default_dtype is None:
            return None
        dtype = typesystem.NumpyDtypeType(dtype=np.dtype(default_dtype))
        return dtype
    else:
        return resolve_attribute_dtype(dtype_arg.variable.type)

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
    return array_from_type(a.variable.type)

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
        return

    return get_dtype(obj)

def empty_like(a, dtype, order):
    "Parse the result type for np.empty_like calls"
    if a is None:
        return

    type = a.variable.type
    if type.is_array:
        if dtype:
            dtype = get_dtype(dtype)
            if dtype is None:
                return type
            dtype = dtype.resolve()
        else:
            dtype = type.dtype

        return minitypes.ArrayType(dtype, type.ndim)

register_inferer(np, 'empty_like', empty_like)
register_inferer(np, 'zeros_like', empty_like)
register_inferer(np, 'ones_like', empty_like)

def empty(shape, dtype, order):
    if shape is None:
        return None

    dtype = get_dtype(dtype, np.float64)
    shape_type = shape.variable.type

    if shape_type.is_int:
        ndim = 1
    elif shape_type.is_tuple or shape_type.is_list:
        ndim = shape_type.size
    else:
        return None

    return minitypes.ArrayType(dtype.resolve(), ndim)

register_inferer(np, 'empty', empty)
register_inferer(np, 'zeros', empty)
register_inferer(np, 'ones', empty)

@register(np)
def arange(start, stop, step, dtype):
    "Resolve a call to np.arange()"
    # NOTE: dtype must be passed as a keyword argument, or as the fourth
    # parameter
    dtype = get_dtype(dtype, np.int64)
    if dtype is not None:
        # return a 1D array type of the given dtype
        return dtype.resolve()[:]

@register(np)
def dot(context, a, b, out):
    "Resolve a call to np.dot()"
    if out is not None:
        return out.variable.type

    lhs_type = promote_to_array(a.variable.type)
    rhs_type = promote_to_array(b.variable.type)

    dtype = context.promote_types(lhs_type.dtype, rhs_type.dtype)
    dst_ndim = lhs_type.ndim + rhs_type.ndim - 2

    result_type = minitypes.ArrayType(dtype, dst_ndim, is_c_contig=True)
    return result_type

@register(np)
def array(object, dtype, order, subok):
    type = array_from_object(object)
    if dtype is not None:
        type = type.copy(dtype=dtype.variable.type)

    return type

@register(np)
def nonzero(a):
    return _nonzero(array_from_object(a))

def _nonzero(type):
    if type.is_array:
        return typesystem.TupleType(index_array_t, type.ndim)
    else:
        return typesystem.TupleType(index_array_t)

@register(np)
def where(context, condition, x, y):
    if x is None and y is None:
        return nonzero(condition)

    xtype = array_from_object(x)
    ytype = array_from_object(y)
    type = context.promote_types(xtype, ytype)
    return type
