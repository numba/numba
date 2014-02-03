# -*- coding: utf-8 -*-

"""
Default rules for the typing of constants.
"""

from __future__ import print_function, division, absolute_import

import math
import types
import ctypes
from functools import partial

import numba.typesystem
from numba.typesystem import itypesystem, numpy_support
from numba import numbawrapper

from numba.support.ctypes_support import is_ctypes, from_ctypes_value
from numba.support import cffi_support

import numpy as np
import datetime

#------------------------------------------------------------------------
# Class -> Type
#------------------------------------------------------------------------

def get_typing_defaults(u):
    """
    Get a simple table mapping Python classes to types.

    :param u: The type universe
    """
    typing_defaults = {
        float: u.double,
        bool: u.bool_,
        complex: u.complex128,
        str: u.string_,
        #datetime.datetime: u.datetime,
        np.datetime64: u.datetime(),
        np.timedelta64: u.timedelta(),
    }
    return typing_defaults

#------------------------------------------------------------------------
# Class -> pyval -> Type
#------------------------------------------------------------------------

def get_default_typing_rules(u, typeof, promote):
    """
    Get a table mapping Python classes to handlers (value -> type)

    :param u: The type universe
    """

    table = {}
    def register(*classes):
        def dec(func):
            for cls in classes:
                table[cls] = lambda u, value: func(value)
            return func
        return dec

    @register(int, long)
    def type_int(value):
        if abs(value) < 1:
            bits = 0
        else:
            bits = math.ceil(math.log(abs(value), 2))

        if bits < 32:
            return u.int_
        elif bits < 64:
            return u.int64
        else:
            raise ValueError("Cannot represent %s as int32 or int64", value)

    @register(np.ndarray)
    def type_ndarray(value):
        if isinstance(value, np.ndarray):
            dtype = numpy_support.map_dtype(value.dtype)
            return u.array(dtype, value.ndim)
                           #is_c_contig=value.flags['C_CONTIGUOUS'],
                           #is_f_contig=value.flags['F_CONTIGUOUS'])

    @register(tuple, list, dict)
    def type_container(value):
        assert isinstance(value, (tuple, list, dict))

        if isinstance(value, dict):
            key_type = type_container(value.keys())
            value_type = type_container(value.values())
            return u.dict_(key_type, value_type, size=len(value))

        if isinstance(value, tuple):
            container_type = u.tuple_
        else:
            container_type = u.list_

        if 0 < len(value) < 30:
            # Figure out base type if the container is not too large
            # base_type = reduce(promote, (typeof(child) for child in value))
            ty = typeof(value[0])
            if all(typeof(child) == ty for child in value):
                base_type = ty
            else:
                base_type = u.object_
        else:
            base_type = u.object_

        return container_type(base_type, size=len(value))

    register(np.dtype)(lambda value: u.numpy_dtype(numpy_support.map_dtype(value)))
    register(types.ModuleType)(lambda value: u.module(value))
    register(itypesystem.Type)(lambda value: u.meta(value))

    return table

def get_constant_typer(universe, typeof, promote):
    """
    Get a function mapping values to types, which returns None if unsuccessful.
    """
    typetable = get_typing_defaults(universe)
    handler_table = get_default_typing_rules(universe, typeof, promote)
    return itypesystem.ConstantTyper(universe, typetable, handler_table).typeof

#------------------------------------------------------------------------
# Constant matching ({ pyval -> bool : pyval -> Type })
#------------------------------------------------------------------------

# TODO: Make this a well-defined (easily overridable) matching table
# E.g. { "numpy" : { is_numpy : get_type } }

def is_dtype_constructor(value):
    return isinstance(value, type) and issubclass(value, np.generic)

def is_numpy_scalar(value): 
    return isinstance(value, np.generic)

def is_registered(value):
    from numba.type_inference import module_type_inference
    return module_type_inference.is_registered(value)

def from_ctypes(value, u):
    result = from_ctypes_value(value)
    if result.is_function:
        pointer = ctypes.cast(value, ctypes.c_void_p).value
        return u.pointer_to_function(value, pointer, result)
    else:
        return result

def from_cffi(value, u):
    signature = cffi_support.get_signature(value)
    pointer = cffi_support.get_pointer(value)
    return u.pointer_to_function(value, pointer, signature)

def from_typefunc(value, u):
    from numba.type_inference import module_type_inference
    result = module_type_inference.module_attribute_type(value)
    if result is not None:
        return result
    else:
        return u.known_value(value)

is_numba_exttype = lambda value: hasattr(type(value), '__numba_ext_type')
is_NULL = lambda value: value is numba.NULL
is_autojit_func = lambda value: isinstance(
    value, numbawrapper.NumbaSpecializingWrapper)

def get_default_match_table(u):
    """
    Get a matcher table: { (type -> bool) : (value -> type) }
    """
    table = {
        is_NULL:
            lambda value: numba.typesystem.null,
        is_dtype_constructor:
            lambda value: numba.typesystem.from_numpy_dtype(np.dtype(value)),
        is_numpy_scalar:
            lambda value: numpy_support.map_dtype(value.dtype),
        is_ctypes:
            lambda value: from_ctypes(value, u),
        cffi_support.is_cffi_func:
            lambda value: from_cffi(value, u),
        is_numba_exttype:
            lambda value: getattr(type(value), '__numba_ext_type'),
        numbawrapper.is_numba_wrapper:
            lambda value: u.jit_function(value),
        is_autojit_func:
            lambda value: u.autojit_function(value),
        is_registered:
            lambda value: from_typefunc(value, u),
    }

    return table

def find_match(matchtable, value):
    for matcher, typefunc in matchtable.iteritems():
        if matcher(value):
            result = typefunc(value)
            assert result is not None
            return result

    return None

#------------------------------------------------------------------------
# Typeof
#------------------------------------------------------------------------

def object_typer(universe, value):
    return universe.object_

def find_first(callables, value):
    for callable in callables:
        result = callable(value)
        if result is not None:
            return result

    assert False, (callables, value)

def get_default_typeof(universe, promote):
    typeof1 = get_constant_typer(universe, lambda value: typeof(value), promote)
    typeof2 = partial(find_match, get_default_match_table(universe))
    typeof3 = partial(object_typer, universe)
    typeof = partial(find_first, [typeof1, typeof2, typeof3])
    return typeof
