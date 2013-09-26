# -*- coding: utf-8 -*-
"""
Type inference for NumPy binary ufuncs and their methods.
"""
from __future__ import print_function, division, absolute_import

import numpy as np

from numba import *
from numba import typesystem
from numba.typesystem import numpy_support
from numba.type_inference.module_type_inference import (module_registry,
                                                        register,
                                                        register_inferer,
                                                        register_unbound)
from numba.typesystem import get_type
from numba.type_inference.modules.numpymodule import (get_dtype,
                                                      array_from_type,
                                                      promote,
                                                      promote_to_array,
                                                      demote_to_scalar)

#----------------------------------------------------------------------------
# Utilities
#----------------------------------------------------------------------------

def array_of_dtype(a, dtype, static_dtype, out):
    if out is not None:
        return out

    a = array_from_type(a)
    if a.is_array:
        dtype = _dtype(a, dtype, static_dtype)
        if dtype is not None:
            return a.add('dtype', dtype)

def _dtype(a, dtype, static_dtype):
    if static_dtype:
        return static_dtype
    elif dtype:
        return dtype.dtype
    elif a.is_array:
        return a.dtype
    elif not a.is_object:
        return a
    else:
        return None

#------------------------------------------------------------------------
# Ufunc Type Strings
#------------------------------------------------------------------------

def numba_type_from_sig(ufunc_signature):
    """
    Convert ufunc type signature string (e.g. 'dd->d') to a function
    """
    args, ret = ufunc_signature.split('->')
    to_numba = lambda c: numpy_support.map_dtype(np.dtype(c))

    signature = to_numba(ret)(*map(to_numba, args))
    return signature

def find_signature(args, signatures):
    for signature in signatures:
        if signature.args == args:
            return signature

def find_ufunc_signature(typesystem, argtypes, signatures):
    """
    Map (float_, double) and [double(double, double),
                              int_(int_, int_),
                              ...]
    to double(double, double)
    """
    signature = find_signature(tuple(argtypes), signatures)

    if signature is not None:
        return signature

    argtype = reduce(typesystem.promote, argtypes)
    if not argtype.is_object:
        args = (argtype,) * len(argtypes)
        return find_signature(args, signatures)

    return None

class UfuncTypeInferer(object):
    "Infer types for arbitrary ufunc"

    def __init__(self, ufunc):
        self.ufunc = ufunc
        self.signatures = set(map(numba_type_from_sig, ufunc.types))

    def infer(self, typesystem, argtypes):
        signature = find_ufunc_signature(typesystem, argtypes, self.signatures)
        if signature is None:
            return None
        else:
            return signature.return_type

def register_arbitrary_ufunc(ufunc):
    "Type inference for arbitrary ufuncs"
    ufunc_infer = UfuncTypeInferer(ufunc)

    def infer(typesystem, *args, **kwargs):
        if len(args) != ufunc.nin:
            return object_

        # Find the right ufunc signature
        argtypes = [type.dtype if type.is_array else type for type in args]
        result_type = ufunc_infer.infer(typesystem, argtypes)

        if result_type is None:
            return object_

        # Determine output ndim
        ndim = 0
        for argtype in args:
            if argtype.is_array:
                ndim = max(argtype.ndim, ndim)

        return typesystem.array(result_type, ndim)

    module_registry.register_value(ufunc, infer)
    # module_registry.register_unbound_dotted_value

#----------------------------------------------------------------------------
# Ufunc type inference
#----------------------------------------------------------------------------

def binary_map(typesystem, a, b, out):
    if out is not None:
        return out

    return promote(typesystem, a, b)

def binary_map_bool(typesystem, a, b, out):
    type = binary_map(typesystem, a, b, out)
    if type.is_array:
        return type.add('dtype', bool_)
    elif type.is_numeric:
        return bool_
    else:
        return object_

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
    return demote_to_scalar(array_of_dtype(a, dtype, static_dtype, out))

def accumulate_bool(a, axis, dtype, out):
    return accumulate(a, axis, dtype, out, bool_)

def reduceat(a, indices, axis, dtype, out, static_dtype=None):
    return accumulate(a, axis, dtype, out, static_dtype)

def reduceat_bool(a, indices, axis, dtype, out):
    return reduceat(a, indices, axis, dtype, out, bool_)

def outer(typesystem, a, b, static_dtype=None):
    a = array_of_dtype(a, None, static_dtype, out=None)
    if a and a.is_array:
        return a.dtype[:, :]

def outer_bool(typesystem, a, b):
    return outer(typesystem, a, b, bool_)

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
)

if not PY3:
    binary_ufuncs_arithmetic = binary_ufuncs_arithmetic + ('divide', )

#------------------------------------------------------------------------
# Register our type functions
#------------------------------------------------------------------------

register_inferer(np, 'sum', reduce_)
register_inferer(np, 'prod', reduce_)

def register_arithmetic_ufunc(register_inferer, register_unbound, binary_ufunc):
    register_inferer(np, binary_ufunc, binary_map)
    register_unbound(np, binary_ufunc, "reduce", reduce_)
    register_unbound(np, binary_ufunc, "accumulate", accumulate)
    register_unbound(np, binary_ufunc, "reduceat", reduceat)
    register_unbound(np, binary_ufunc, "outer", outer)

def register_bool_ufunc(register_inferer, register_unbound, binary_ufunc):
    register_inferer(np, binary_ufunc, binary_map_bool)
    register_unbound(np, binary_ufunc, "reduce", reduce_bool)
    register_unbound(np, binary_ufunc, "accumulate", accumulate_bool)
    register_unbound(np, binary_ufunc, "reduceat", reduceat_bool)
    register_unbound(np, binary_ufunc, "outer", outer_bool)

for binary_ufunc in binary_ufuncs_bitwise + binary_ufuncs_arithmetic:
    register_arithmetic_ufunc(register_inferer, register_unbound, binary_ufunc)

for binary_ufunc in binary_ufuncs_compare + binary_ufuncs_logical:
    register_bool_ufunc(register_inferer, register_unbound, binary_ufunc)
