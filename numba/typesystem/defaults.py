# -*- coding: utf-8 -*-

"""
Type defaults
"""

from __future__ import print_function, division, absolute_import

import math

from numba.typesystem import universe
from numba.typesystem.typesystem import (
    Universe, Type, Conser, nbo, ConstantTyper, TypeConverter, TypeSystem)
from numba.typesystem import promotion
from numba.typesystem import constants
from numba.typesystem import lowering

import numpy as np

#------------------------------------------------------------------------
# Defaults initialization
#------------------------------------------------------------------------

def compose(f, g):
    return lambda x: f(g(x))

# ______________________________________________________________________
# Universes

numba_universe = universe.NumbaUniverse()
llvm_universe = universe.LLVMUniverse()

# ______________________________________________________________________
# Converters

default_type_lowerer = lowering.create_type_lowerer(
    lowering.default_numba_lowering_table, numba_universe, numba_universe)
to_llvm_converter = TypeConverter(numba_universe, llvm_universe)

lower = default_type_lowerer.convert
to_llvm = to_llvm_converter.convert

converters = {
    "llvm": compose(to_llvm, lower),
}

# ______________________________________________________________________
# Typesystems

promote = promotion.get_default_promoter(numba_universe)
typeof = constants.get_default_typeof(numba_universe, promote)

numba_typesystem = TypeSystem(numba_universe, promote, typeof, converters)
llvm_typesystem = TypeSystem(llvm_universe, typeof=compose(lower, typeof))

# ______________________________________________________________________

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
