# -*- coding: utf-8 -*-

"""
Type defaults
"""

from __future__ import print_function, division, absolute_import

import math

from numba.typesystem import universe
from numba.typesystem.typesystem import (
    Universe, Type, Conser, nbo, ConstantTyper, TypeConverter, TypeSystem)
from numba.typesystem.usertypes import *

import numpy as np


def find_type_of_size(size, typelist):
    for type in typelist:
        if type.itemsize == size:
            return type

    assert False, "Type of size %d not found: %s" % (size, typelist)


class DefaultConstantTyper(ConstantTyper):

    def typeof(self, value):
        u = self.universe

        if isinstance(value, float):
            return u.double
        elif isinstance(value, bool):
            return u.bool
        elif isinstance(value, (int, long)):
            if abs(value) < 1:
                bits = 0
            else:
                bits = math.ceil(math.log(abs(value), 2))

            if bits < 32:
                return u.int
            elif bits < 64:
                return u.int64
            else:
                raise ValueError("Cannot represent %s as int32 or int64", value)
        elif isinstance(value, complex):
            return u.complex128
        elif isinstance(value, str):
            return u.c_string_type
        elif isinstance(value, np.ndarray):
            from numba.support import numpy_support
            dtype = numpy_support.map_dtype(value.dtype)
            return ArrayType(dtype, value.ndim,
                             is_c_contig=value.flags['C_CONTIGUOUS'],
                             is_f_contig=value.flags['F_CONTIGUOUS'])
        else:
            return u.object


class DefaultUnifier(object):

    def __init__(self, universe):
        self.universe = universe

    def promote_numeric(self, type1, type2):
        "Promote two numeric types"
        u = self.universe

        type = max([type1, type2], key=lambda type: type.rank)
        if type1.kind != type2.kind:
            def itemsize(type):
                size = u.itemsize(type)
                return size // 2 if type.is_complex else size

            size = max(itemsize(type1), itemsize(type2))
            if type.is_complex:
                type = find_type_of_size(size * 2, complextypes)
            elif type.is_float:
                type = find_type_of_size(size, floating)
            else:
                assert type.is_int
                type = find_type_of_size(size, integral)

        return type

    def promote_arrays(self, type1, type2):
        "Promote two array types in an expression to a new array type"
        u = self.universe

        equal_ndim = type1.ndim == type2.ndim
        return u.array(self.unify(type1.dtype, type2.dtype),
                       ndim=max((type1.ndim, type2.ndim)),
                       is_c_contig=(equal_ndim and type1.is_c_contig and
                                    type2.is_c_contig),
                       is_f_contig=(equal_ndim and type1.is_f_contig and
                                    type2.is_f_contig))

    def unify(self, type1, type2):
        "Promote two arbitrary types"
        u = self.universe

        string_types = u.c_string_type, u.char.pointer()

        if type1.is_pointer and type2.is_int_like:
            return type1
        elif type2.is_pointer and type2.is_int_like:
            return type2
        elif type1.is_object or type2.is_object:
            return u.object_
        elif type1.is_numeric and type2.is_numeric:
            return self.promote_numeric(type1, type2)
        elif type1.is_array and type2.is_array:
            return self.promote_arrays(type1, type2)
        elif type1 in string_types and type2 in string_types:
            return u.c_string_type
        elif type1.is_bool and type2.is_bool:
            return u.bool_
        else:
            raise TypeError((type1, type2))


numba_universe = universe.NumbaUniverse()
llvm_universe = universe.LLVMUniverse()

default_converters = {
    "llvm": TypeConverter(numba_universe, llvm_universe),
}

numba_typesystem = TypeSystem(
    numba_universe,
    DefaultUnifier(numba_universe),
    DefaultConstantTyper(numba_universe),
    default_converters)

llvm_typesystem = TypeSystem(llvm_universe)