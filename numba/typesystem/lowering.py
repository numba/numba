# -*- coding: utf-8 -*-

"""
Type lowering from a higher-level domain to a lower-level domain.
"""

from __future__ import print_function, division, absolute_import

from functools import partial

from numba.typesystem import typesystem
from numba.typesystem.kinds import *

def create_type_lowerer(table, domain, codomain):
    """
    Create a type lowerer from a domain to a codomain given a lowering table.
    """
    def convert_mono(domain, codomain, type):
        ctor = table.get(type.name, typesystem.convert_mono)
        return ctor(domain, codomain, type)

    def convert_poly(domain, codomain, type, params):
        ctor = table.get(type.kind, typesystem.convert_poly)
        return ctor(domain, codomain, type, params)

    return typesystem.TypeConverter(domain, codomain, convert_mono, convert_poly)

#------------------------------------------------------------------------
# Lowering functions
#------------------------------------------------------------------------

def lower_complex(domain, codomain, type, params):
    base_type, = params
    return codomain.struct([('real', base_type), ('imag', base_type)])

#------------------------------------------------------------------------
# Default Lowering Table
#------------------------------------------------------------------------

default_lowering_table = {
    KIND_COMPLEX: lower_complex,
}

# default_lowerer = create_type_lowerer(
#     default_lowering_table, numba_universe, numba_universe)