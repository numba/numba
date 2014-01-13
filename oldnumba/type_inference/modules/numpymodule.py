# -*- coding: utf-8 -*-
"""
Infer types for NumPy functionality. This includes:

    1) Figuring out dtypes

        e.g. np.double     -> double
             np.dtype('d') -> double

    2) Function calls such as np.empty/np.empty_like/np.arange/etc
"""
from __future__ import print_function, division, absolute_import

import warnings
from functools import reduce

import numpy as np

from numba import *
from numba import typesystem, error
from numba.type_inference.module_type_inference import (register,
                                                        register_inferer,
                                                        register_unbound)
from numba.typesystem import get_type


#------------------------------------------------------------------------
# Type Definitions
#------------------------------------------------------------------------

index_array_t = npy_intp[:]

#------------------------------------------------------------------------
# Some utilities
#------------------------------------------------------------------------

def promote(typesystem, *types):
    return reduce(typesystem.promote, map(array_from_type, types))

def resolve_attribute_dtype(dtype, default=None):
    "Resolve the type for numpy dtype attributes"
    if dtype.is_numpy_dtype:
        return dtype

    if dtype.is_known_value:
        numpy_attr = dtype.value
        if isinstance(numpy_attr, np.dtype):
            return typesystem.from_numpy_dtype(numpy_attr)
        elif issubclass(numpy_attr, np.generic):
            return typesystem.from_numpy_dtype(np.dtype(numpy_attr))
        elif numpy_attr is not None:
            try:
                dtype = np.dtype(numpy_attr)
            except TypeError:
                warnings.warn("Unable to infer dtype for '%s'" % numpy_attr)
            else:
                return typesystem.from_numpy_dtype(dtype)

    return None

def get_dtype(dtype_arg, default_dtype=None):
    """
    Simple helper function to map an AST node dtype keyword
    argument => NumPy dtype.
    '"""
    if dtype_arg is None:
        if default_dtype is None:
            return None

        return typesystem.numpy_dtype(default_dtype)
    else:
        return resolve_attribute_dtype(dtype_arg)

def promote_to_array(dtype):
    "Promote scalar to 0d array type"
    if not dtype.is_array:
        dtype = typesystem.array_(dtype, 0)
    return dtype

def demote_to_scalar(type):
    "Demote 0d arrays to scalars"
    if type and type.is_array and type.ndim == 0:
        return type.dtype
    return type

def array_from_object(a):
    """
    object -> array type:

        array_from_object(ASTNode([[1, 2], [3, 4]])) => int64[:, :]
    """
    return array_from_type(get_type(a))

def array_from_type(type):
    if type.is_array:
        return type
    elif type.is_tuple or type.is_list:
        dtype = array_from_type(type.base_type)
        if dtype.is_array:
            return dtype.add('ndim', dtype.ndim + 1)
    elif not type.is_object:
        return typesystem.array_(dtype=type, ndim=0)

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
    if dtype is None:
        return object_

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
    dtype_type = get_dtype(dtype, npy_intp)
    if dtype_type is not None:
        # return a 1D array type of the given dtype
        return dtype_type.dtype[:]

@register(np)
def dot(typesystem, a, b, out):
    "Resolve a call to np.dot()"
    if out is not None:
        return out

    lhs_type = promote_to_array(a)
    rhs_type = promote_to_array(b)

    dtype = typesystem.promote(lhs_type.dtype, rhs_type.dtype)
    dst_ndim = lhs_type.ndim + rhs_type.ndim - 2

    result_type = typesystem.array(dtype, dst_ndim)
    return result_type

@register(np)
def array(object, dtype, order, subok):
    type = array_from_type(object)
    if type.is_array and dtype is not None:
        type = type.add('dtype', dtype)
    elif dtype is not None:
        return dtype
    else:
        return type

@register(np, pass_in_types=False)
def datetime64(datetime_string):
    return nodes.NumpyDateTimeNode(datetime_string)

@register(np, pass_in_types=False)
def timedelta64(delta, units):
    return nodes.NumpyTimeDeltaNode(delta, units)

@register(np)
def nonzero(a):
    return _nonzero(array_from_type(a))

def _nonzero(type):
    if type.is_array:
        return typesystem.tuple_(index_array_t, type.ndim)
    else:
        return typesystem.tuple_(index_array_t)

@register(np)
def where(typesystem, condition, x, y):
    if x is None and y is None:
        return nonzero(condition)

    return promote(typesystem, x, y)

@register(np)
def vdot(typesystem, a, b):
    lhs_type = promote_to_array(a)
    rhs_type = promote_to_array(b)
    dtype = typesystem.promote(lhs_type.dtype, rhs_type.dtype)
    return dtype

@register(np)
def inner(typesystem, a, b):
    lhs_type = promote_to_array(a)
    rhs_type = promote_to_array(b)
    dtype = typesystem.promote(lhs_type.dtype, rhs_type.dtype)
    if lhs_type.ndim == 0:
        result_ndim = rhs_type.ndim
    elif rhs_type.ndim == 0:
        result_ndim = lhs_type.ndim
    else:
        result_ndim = lhs_type.ndim + rhs_type.ndim - 2
    if result_ndim == 0:
        result_type = dtype
    else:
        result_type = typesystem.array(dtype, result_ndim)
    return result_type

@register(np)
def outer(typesystem, a, b):
    result_type = promote(typesystem, a, b)
    # promote() converts scalar types to 0-dim arrays, so it should
    # always return an array type.  Ensure this continues to hold...
    assert result_type.is_array
    return result_type.dtype[:, :]

@register(np, pass_in_types=False)
def tensordot(typesystem, a, b, axes):
    '''Typing function for numpy.tensordot().

    Defaults to Python object for any caller that isn't using the
    default argument to axes.

    Otherwise, it is similar to inner(), but subtracts four dimensions
    from the result instead of two.

    Without symbolic execution of the actual axes argument, this can't
    determine the number of axes to sum over, so it punts.  This
    typing function could use an array type of unknown dimensionality,
    were one available.  See:
    https://www.pivotaltracker.com/story/show/43687249
    '''
    lhs_type = array_from_object(a)
    rhs_type = array_from_object(b)
    if lhs_type.ndim < 1:
        raise error.NumbaError(a, 'First argument to numpy.tensordot() '
                               'requires array of dimensionality >= 1.')
    elif rhs_type.ndim < 1:
        raise error.NumbaError(b, 'First argument to numpy.tensordot() '
                               'requires array of dimensionality >= 1.')
    dtype = typesystem.promote(lhs_type.dtype, rhs_type.dtype)
    if axes is None:
        result_ndim = lhs_type.ndim + rhs_type.ndim - 4
        if result_ndim < 0:
            raise error.NumbaError(a, 'Arguments to numpy.tensordot() should '
                                   'have combined dimensionality >= 4 (when '
                                   'axes argument is not specified).')
        result_type = typesystem.array(dtype, result_ndim)
    else:
        # XXX Issue warning to user?
        result_type = object_
    return result_type

@register(np)
def einsum(typesystem, subs, operands, kws):
    # XXX Issue warning to user?
    # XXX Attempt type inference in case where subs is a string?
    return object_

@register(np)
def kron(typesystem, a, b):
    #raise NotImplementedError("XXX")
    return object_

@register(np)
def trace(typesystem, a, offset, axis1, axis2, dtype, out):
    #raise NotImplementedError("XXX")
    return object_

#------------------------------------------------------------------------
# numpy.linalg
#------------------------------------------------------------------------

@register(np.linalg)
def cholesky(typesystem, a):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def cond(typesystem, x, p):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def det(typesystem, a):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def eig(typesystem, a):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def eigh(typesystem, a, UPLO):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def eigvals(typesystem, a):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def eigvalsh(typesystem, a, UPLO):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def inv(typesystem, a):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def lstsq(typesystem, a, b, rcond):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def matrix_power(typesystem, M, n):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def matrix_rank(typesystem, M, tol):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def norm(typesystem, x, ord):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def pinv(typesystem, a, rcond):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def qr(typesystem, a, mode):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def slogdet(typesystem, a):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def solve(typesystem, a, b):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def svd(typesystem, a, full_matrices, compute_uv):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def tensorinv(typesystem, a, ind):
    #raise NotImplementedError("XXX")
    return object_

@register(np.linalg)
def tensorsolve(typesystem, a, b, axes):
    #raise NotImplementedError("XXX")
    return object_
