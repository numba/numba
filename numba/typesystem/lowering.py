# -*- coding: utf-8 -*-

"""
Type lowering from a higher-level domain to a lower-level domain.
"""

from __future__ import print_function, division, absolute_import

from numba.typesystem import typesystem
from numba.typesystem.kinds import *

def create_type_lowerer(table, domain, codomain):
    """
    Create a type lowerer from a domain to a codomain given a lowering table.
    """
    def convert_mono(domain, codomain, type):
        ctor = table.get(type.typename, typesystem.convert_mono)
        return ctor(domain, codomain, type)

    def convert_poly(domain, codomain, type, params):
        ctor = table.get(type.kind, typesystem.convert_poly)
        return ctor(domain, codomain, type, params)

    return typesystem.TypeConverter(domain, codomain, convert_mono, convert_poly)

#------------------------------------------------------------------------
# Lowering functions
#------------------------------------------------------------------------

# ______________________________________________________________________
# mono types

def numba_lower_object(domain, codomain, type):
    assert domain.name == "numba" # These only work in the numba domain
    from numba import typedefs # hurr
    return codomain.pointer(typedefs.PyObject_HEAD)

def numba_lower_array(domain, codomain, type):
    assert domain.name == "numba"
    from numba import typedefs
    return codomain.pointer(typedefs.PyArray)

# ______________________________________________________________________
# poly types

def lower_complex(domain, codomain, type, params):
    base_type, = params
    return codomain.struct([('real', base_type), ('imag', base_type)])

#------------------------------------------------------------------------
# Default Lowering Table
#------------------------------------------------------------------------

default_numba_lowering_table = {
    KIND_COMPLEX: lower_complex,
    KIND_OBJECT: numba_lower_object,
    KIND_ARRAY: numba_lower_array,
}

# default_lowerer = create_type_lowerer(
#     default_lowering_table, numba_universe, numba_universe)