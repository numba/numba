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
from numba.typesystem import numbatypes as numba_domain
from numba.typesystem import llvmtypes as llvm_domain
from numba.typesystem import types

#------------------------------------------------------------------------
# Defaults initialization
#------------------------------------------------------------------------

def compose(f, g):
    return lambda x: f(g(x))

# ______________________________________________________________________
# Converters

default_type_lowerer = lowering.create_type_lowerer(
    lowering.default_numba_lowering_table, numba_domain, numba_domain)
to_llvm_converter = TypeConverter(numba_domain, llvm_domain)

lower = default_type_lowerer.convert
to_llvm = to_llvm_converter.convert

converters = {
    "llvm": compose(to_llvm, lower),
}

# ______________________________________________________________________
# Typesystems

promote = promotion.get_default_promoter(numba_domain)
typeof = constants.get_default_typeof(numba_domain, promote)

numba_typesystem = TypeSystem(numba_domain, promote, typeof, converters)
llvm_typesystem = TypeSystem(llvm_domain, typeof=compose(lower, typeof))
