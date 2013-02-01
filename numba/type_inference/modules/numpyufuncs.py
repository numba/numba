"""
Type inference for NumPy ufuncs.
"""

import numpy as np

from numba import *
from numba.minivect import minitypes
from numba import typesystem
from numba.type_inference.module_type_inference import (register,
                                                        register_inferer,
                                                        register_unbound)
from numba.typesystem import get_type
from numba.type_inference.modules.numpymodule import (get_dtype,
                                                      array_from_type,
                                                      promote,
                                                      promote_to_array)

def _dtype(a, dtype, static_dtype):
    if static_dtype:
        return static_dtype
    elif dtype:
        return dtype
    elif a.is_array:
        return a.dtype
    elif not a.is_object:
        return a
    else:
        return None

def binary_map(context, a, b, out):
    if out is not None:
        return out

    return promote(context, a, b)

def binary_map_bool(context, a, b, out):
    type = binary_map(context, a, b, out)
    if type.is_array:
        return type.copy(dtype=bool_)
    else:
        return bool_

def reduce_(a, axis, dtype, out, static_dtype=None):
    if out is not None:
        return out

    dtype_type = _dtype(a, dtype, static_dtype)

    if axis is None:
        # Return the scalar type
        return dtype_type

    if dtype_type:
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

def reduce_bool(a, axis, dtype, out):
    return reduce_(a, axis, dtype, out, bool_)

def accumulate(a, axis, dtype, out, static_dtype=None):
    if out is not None:
        return out

    dtype = _dtype(a, dtype, static_dtype)
    if dtype:
        return promote_to_array(a).copy(dtype=dtype)

def accumulate_bool(a, axis, dtype, out):
    return accumulate(a, axis, dtype, out, bool_)

#------------------------------------------------------------------------
# Binary Ufuncs
#------------------------------------------------------------------------

binary_ufuncs_compare = (
    # Comparisons
    'greater',
    'greater_equal',
    'less',
    'less_equal',
    'not_equal',
    'equal',
)

binary_ufuncs_logical = (
    # Logical ufuncs
    'logical_and',
    'logical_or',
    'logical_xor',
    'logical_not',
)

binary_ufuncs_bitwise = (
    # Bitwise ufuncs
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'left_shift',
    'right_shift',
)

binary_ufuncs_arithmetic = (
    # Arithmetic ufuncs
    'add',
    'subtract',
    'multiply',
    'true_divide',
    'floor_divide',
    'divide',
)

#------------------------------------------------------------------------
# Register our type functions
#------------------------------------------------------------------------

register_inferer(np, 'sum', reduce_)
register_inferer(np, 'prod', reduce_)

for binary_ufunc in binary_ufuncs_bitwise + binary_ufuncs_arithmetic:
    register_inferer(np, binary_ufunc, binary_map)
    register_unbound(np, binary_ufunc, "reduce", reduce_)
    register_unbound(np, binary_ufunc, "accumulate", accumulate)

for binary_ufunc in binary_ufuncs_compare + binary_ufuncs_logical:
    register_inferer(np, binary_ufunc, binary_map_bool)
    register_unbound(np, binary_ufunc, "reduce", reduce_bool)
    register_unbound(np, binary_ufunc, "accumulate", accumulate_bool)
