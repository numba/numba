# -*- coding: utf-8 -*-

"""
Type defaults
"""

from __future__ import print_function, division, absolute_import

from numba.typesystem import universe
from numba.typesystem.typesystem import TypeConverter, TypeSystem
from numba.typesystem import promotion
from numba.typesystem import constants
from numba.typesystem import lowering
from numba.typesystem import types

#------------------------------------------------------------------------
# Defaults initialization
#------------------------------------------------------------------------

def compose(f, g):
    return lambda x: f(g(x))

# ______________________________________________________________________
# Universes

lowlevel_universe = universe.LowLevelUniverse()
numba_universe = universe.NumbaUniverse()
llvm_universe = universe.LLVMUniverse()

# ______________________________________________________________________
# Converters

default_type_lowerer = lowering.create_type_lowerer(
    lowering.default_numba_lowering_table, numba_universe, lowlevel_universe)
to_llvm_converter = TypeConverter(lowlevel_universe, llvm_universe)

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

u = numba_universe

integral = []
native_integral = []
floating = []
complextypes = [u.complex64, u.complex128, u.complex256]

for typename, ty in u.monotypes.iteritems():
    if ty.is_int:
        integral.append(ty)
    elif ty.is_float:
        floating.append(ty)

numeric = integral + floating + complextypes
native_integral.extend((u.Py_ssize_t, u.size_t))

# integral.sort(key=types.sort_types_key)
# native_integral = [ty for ty in integral if ty.typename in universe.native_sizes]
# floating.sort(key=types.sort_types_key)
# complextypes.sort(key=types.sort_types_key)
