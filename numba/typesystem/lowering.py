# -*- coding: utf-8 -*-

"""
Type lowering from a higher-level domain to a lower-level domain.
"""

from __future__ import print_function, division, absolute_import

from numba.typesystem import typesystem

def create_type_lowerer(table, domain, codomain):
    """
    Create a type lowerer from a domain to a codomain given a lowering table.
    """
    def convert_mono(domain, codomain, type):
        if type.typename in table:
            return table[type.typename](domain, codomain, type, ())
        else:
            return typesystem.convert_mono(domain, codomain, type)

    def convert_poly(domain, codomain, type, params):
        ctor = table.get(type.kind, typesystem.convert_poly)
        return ctor(domain, codomain, type, params)

    return typesystem.TypeConverter(domain, codomain, convert_mono, convert_poly)

#------------------------------------------------------------------------
# Lowering functions
#------------------------------------------------------------------------

# ______________________________________________________________________
# mono types

def numba_lower_object(domain, codomain, type, params):
    from numba import typedefs # hurr
    return codomain.pointer(typedefs.PyObject_HEAD)

def numba_lower_array(domain, codomain, type, params):
    from numba import typedefs
    return codomain.pointer(typedefs.PyArray)

# ______________________________________________________________________
# poly types

def lower_complex(domain, codomain, type, params):
    base_type, = params
    return codomain.struct_([('real', base_type), ('imag', base_type)])

#------------------------------------------------------------------------
# Default Lowering Table
#------------------------------------------------------------------------

default_numba_lowering_table = {
    "complex_":         lower_complex,
    "object_":          numba_lower_object,
    "extension":        numba_lower_object,
    "jit_exttype":      numba_lower_object,
    "autojit_exttype":  numba_lower_object,
    "array":            numba_lower_array,
}