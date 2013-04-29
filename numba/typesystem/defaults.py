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
from numba.typesystem import constants

import numpy as np

#------------------------------------------------------------------------
# Typing of Constants
#------------------------------------------------------------------------

class DefaultConstantTyper(object):

    def __init__(self, universe):
        ts = numba_typesystem
        self.typer = constants.get_constant_typer(
            universe, ts.typeof, ts.promote)

        # TODO: hurr
        self.matchtable = constants.get_default_match_table(numba_typesystem)

    def typeof(self, value):
        result = self.typer.typeof(value)
        if result is not None:
            return result

        for matcher, typefunc in self.matchtable.iteritems():
            if matcher(value):
                result = typefunc(value)
                assert result is not None
                return result

        return self.universe.object

#------------------------------------------------------------------------
# Type Promotion
#------------------------------------------------------------------------

def find_type_of_size(size, typelist):
    for type in typelist:
        if type.itemsize == size:
            return type

    assert False, "Type of size %d not found: %s" % (size, typelist)


class DefaultPromoter(object):

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
    DefaultPromoter(numba_universe),
    None,
    default_converters)

numba_typesystem.constant_typer = DefaultConstantTyper(numba_universe)

llvm_typesystem = TypeSystem(llvm_universe)

integral = []
native_integral = []
floating = []
complextypes = []

# for typename in __all__:
#     minitype = globals()[typename]
#     if minitype is None:
#         continue
#
#     if minitype.is_int:
#         integral.append(minitype)
#     elif minitype.is_float:
#         floating.append(minitype)
#     elif minitype.is_complex:
#         complextypes.append(minitype)
#
# numeric = integral + floating + complextypes
# native_integral.extend((Py_ssize_t, size_t))
#
# integral.sort(key=_sort_types_key)
# native_integral = [minitype for minitype in integral
#                                 if minitype.typecode is not None]
# floating.sort(key=_sort_types_key)
# complextypes.sort(key=_sort_types_key)
