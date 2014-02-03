# -*- coding: utf-8 -*-

"""
Type defaults
"""

from __future__ import print_function, division, absolute_import

from . itypesystem import TypeConverter, TypeSystem
from . import promotion, constants, lowering
from . import numbatypes as numba_domain
from . import llvmtypes as llvm_domain
from . import ctypestypes as ctypes_domain

#------------------------------------------------------------------------
# Defaults initialization
#------------------------------------------------------------------------

def compose(f, g):
    return lambda x: f(g(x))

# ______________________________________________________________________
# Converters

def lowerer(table):
    return lowering.create_type_lowerer(table, numba_domain, numba_domain)

# Lowerers
lower = lowerer(lowering.default_numba_lowering_table).convert
ctypes_lower = lowerer(lowering.ctypes_lowering_table).convert

# Converters
to_llvm_converter = TypeConverter(numba_domain, llvm_domain)
to_ctypes_converter = TypeConverter(numba_domain, ctypes_domain)

# ...
to_llvm = to_llvm_converter.convert
to_ctypes = to_ctypes_converter.convert

converters = {
    "llvm": compose(to_llvm, lower),
    "ctypes": compose(to_ctypes, ctypes_lower),
}

# ______________________________________________________________________
# Typesystems

promote = promotion.get_default_promoter(numba_domain)
typeof = constants.get_default_typeof(numba_domain, promote)

numba_typesystem = TypeSystem(numba_domain, promote, typeof, converters)
llvm_typesystem = TypeSystem(llvm_domain, typeof=compose(to_llvm, typeof))
ctypes_typesystem = TypeSystem(ctypes_domain, typeof=compose(to_ctypes, typeof))
