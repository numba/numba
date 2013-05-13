# -*- coding: utf-8 -*-

"""
Type lowering from a higher-level domain to a lower-level domain.
"""

from __future__ import print_function, division, absolute_import
from numba.typesystem import itypesystem

def find_matches(table, flags):
    "Find a lowering function from the flags of the type"
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

def find_func(table, kind, flags, default=None):
    "Get a function form the table by resolving any indirections"
    if kind in table:
        flag = kind
    else:
        flag = find_matches(table, flags)
        if flag is None:
            return default

    while flag in table:
        if isinstance(table[flag], basestring):
            flag = table[flag]
        else:
            return table[flag]

    return default

def create_type_lowerer(table, domain, codomain):
    """
    Create a type lowerer from a domain to a codomain given a lowering table.
    """
    def convert_mono(domain, codomain, type):
        func = find_func(table, type.typename, type.flags)
        if func:
            return func(domain, codomain, type, ())
        else:
            return itypesystem.convert_mono(domain, codomain, type)

    def convert_poly(domain, codomain, type, params):
        ctor = find_func(table, type.kind, type.flags, itypesystem.convert_poly)
        # print("lowering...", type, ctor)
        return ctor(domain, codomain, type, params)

    return itypesystem.TypeConverter(domain, codomain, convert_mono, convert_poly)

#------------------------------------------------------------------------
# Lowering functions
#------------------------------------------------------------------------

# ______________________________________________________________________
# mono types

def lower_object(domain, codomain, type, params):
    from numba import typedefs # hurr
    return codomain.pointer(typedefs.PyObject_HEAD)

def lower_string(domain, codomain, type, params):
    return codomain.pointer(codomain.char)

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
    # print("lowered", type, result, params)
    return result

def lower_extmethod(domain, codomain, type, params):
    return lower_function(domain, codomain, type, params[:4])

def lower_complex(domain, codomain, type, params):
    base_type, = params
    return codomain.struct_([('real', base_type), ('imag', base_type)])

def lower_array(domain, codomain, type, params):
    from numba import typedefs
    return codomain.pointer(typedefs.PyArray)

def lower_to_pointer(domain, codomain, type, params):
    return codomain.pointer(params[0])

def lower_pointer_to_function(domain, codomain, type, params):
    obj, ptr, sig = params
    return sig

#------------------------------------------------------------------------
# Default Lowering Table
#------------------------------------------------------------------------

default_numba_lowering_table = {
    "object":           lower_object,
    # polytypes
    "function":         lower_function,
    "complex":          lower_complex,
    "array":            lower_array,
    "string":           lower_string,
    "carray":           lower_to_pointer,
    "sized_pointer":    lower_to_pointer,
    "reference":        lower_to_pointer,
    "extmethod":        lower_extmethod,
    "pointer_to_function": lower_pointer_to_function,
}

ctypes_lowering_table = {
    "object":           lambda dom, cod, type, params: cod.object_,
    "complex":          lower_complex,
    "array":            "object",
    # "string":           lambda dom, cod, type, params: ctypes.c_char_p,
    "sized_pointer":    lower_to_pointer,
    "reference":        lower_to_pointer,
    "extmethod":        lower_extmethod,
}