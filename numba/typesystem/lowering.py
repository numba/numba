# -*- coding: utf-8 -*-

"""
Type lowering from a higher-level domain to a lower-level domain.
"""

from __future__ import print_function, division, absolute_import

from numba.typesystem import typesystem

def find_matches(flags, table):
    matches = []
    for flag in flags:
        if flag in table:
            matches.append(flag)

    if len(matches) > 1:
        raise ValueError("Multiple matching flags: %s" % flags)
    elif matches:
        return matches[0]
    else:
        return None

def create_type_lowerer(table, domain, codomain):
    """
    Create a type lowerer from a domain to a codomain given a lowering table.
    """
    def convert_mono(domain, codomain, type):
        if type.typename in table:
            return table[type.typename](domain, codomain, type, ())
        else:
            flags = [type.typename] + type.flags
            match = find_matches(flags, table)
            if match:
                return table[type.typename](domain, codomain, type, ())

            return typesystem.convert_mono(domain, codomain, type)

    def convert_poly(domain, codomain, type, params):
        ctor = table.get(type.kind, typesystem.convert_poly)
        # print("lowering...", type, ctor)
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

# ______________________________________________________________________
# poly types

def lower_function(domain, codomain, type, params):
    restype, args, name, is_vararg = params
    newargs = []

    for arg in args:
        if arg.is_struct or arg.is_function:
            arg = codomain.pointer(arg)
        newargs.append(arg)

    if restype.is_struct:
        newargs.append(codomain.pointer(restype))
        restype = codomain.void

    result = codomain.function(restype, newargs, name, is_vararg)
    # print("lowered", type, result)
    return result

def lower_complex(domain, codomain, type, params):
    base_type, = params
    return codomain.struct_([('real', base_type), ('imag', base_type)])

def numba_lower_array(domain, codomain, type, params):
    from numba import typedefs
    return codomain.pointer(typedefs.PyArray)

#------------------------------------------------------------------------
# Default Lowering Table
#------------------------------------------------------------------------

default_numba_lowering_table = {
    "object":           numba_lower_object,
    # polytypes
    "function":         lower_function,
    "complex":          lower_complex,
    "tuple":            numba_lower_object,
    "list":             numba_lower_object,
    "extension":        numba_lower_object,
    "jit_exttype":      numba_lower_object,
    "autojit_exttype":  numba_lower_object,
    "array":            numba_lower_array,
}