# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import copy
import types
import ctypes
from functools import reduce

import llvm.core
import numpy as np

import numba
from numba.support.ctypes_support import is_ctypes, from_ctypes_value
from numba.support import cffi_support
from numba.minivect.minitypes import map_dtype, object_
import numba.minivect.minitypes

from numba import numbawrapper
from numba.typesystem import *
import numba.typesystem

def is_dtype_constructor(value):
    return isinstance(value, type) and issubclass(value, np.generic)


class NumbaTypeMapper(minitypes.TypeMapper):
    """
    Map types from Python values, handle type promotions, etc
    """

    def __init__(self, context):
        super(NumbaTypeMapper, self).__init__(context)
        # self.ctypes_func_type = type(ctypes.CFUNCTYPE(ctypes.c_int))
        # libc = ctypes.CDLL(ctypes.util.find_library('c'))
        # self.ctypes_func_type2 = type(libc.printf)

    def to_llvm(self, type):
        from numba import llvm_types
        if type.is_array:
            return llvm_types._numpy_array
        elif type.is_complex:
            lbase_type = type.base_type.to_llvm(self.context)
            return llvm.core.Type.struct([lbase_type, lbase_type])
        elif type.is_py_ssize_t:
            return llvm_types._llvm_py_ssize_t
        elif type.is_object:
            return llvm_types._pyobject_head_struct_p

        return super(NumbaTypeMapper, self).to_llvm(type)

    def promote_types(self, type1, type2, assignment=False):
        have = lambda p1, p2: have_properties(type1, type2, p1, p2)

        if (type1.is_array or type2.is_array) and not \
            (type1.is_array and type2.is_array):
            if type1.is_array:
                array_type = type1
                other_type = type2
            else:
                array_type = type2
                other_type = type1

            type = copy.copy(array_type)
            type.dtype = self.promote_types(array_type.dtype, other_type)

            if type.dtype.is_object and not array_type.dtype.is_object:
                # Make sure that (double[:], object_) -> object_
                type = object_
            return type
        elif type1.is_unresolved or type2.is_unresolved:
            if type1.is_unresolved:
                type1 = type1.resolve()
            if type2.is_unresolved:
                type2 = type2.resolve()

            if type1.is_unresolved or type2.is_unresolved:
                # The Variable is really only important for ast.Name, fabricate
                # one
                from numba import symtab
                var = symtab.Variable(None)
                return PromotionType(var, self.context, [type1, type2])
            else:
                return self.promote_types(type1, type2)
        elif have("is_pointer", "is_null"):
            return [type1, type2][type1.is_null] # return the pointer type
        elif have("is_pointer", "is_int"):
            return [type1, type2][type1.is_int] # return the pointer type

        return super(NumbaTypeMapper, self).promote_types(type1, type2)


def have_properties(type1, type2, property1, property2):
    """
    Return whether the two types satisfy the two properties:

    >>> have_properties(int32, int32.pointer(), "is_pointer", "is_int")
    True
    """
    type1_p1 = getattr(type1, property1)
    type1_p2 = getattr(type1, property2)
    type2_p1 = getattr(type2, property1)
    type2_p2 = getattr(type2, property2)

    if (type1_p1 and type2_p2) or (type1_p2 and type2_p1):
        if type1_p1:
            return type1
        else:
            return type2
    else:
        return None
