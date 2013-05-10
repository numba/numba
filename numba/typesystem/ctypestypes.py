# -*- coding: utf-8 -*-

"""
ctypes type universe.
"""

from __future__ import print_function, division, absolute_import

import ctypes

from numba.typesystem.itypesystem import consing, tyname
from numba.typesystem import universe
from numba.typesystem import numbatypes as ts

# ______________________________________________________________________

ctypes_map = {
    ts.float32:    ctypes.c_float,
    ts.float64:    ctypes.c_double,
    ts.float128:   ctypes.c_longdouble,
    ts.object_:    ctypes.py_object,
    ts.void:       None,
    ts.string_:    ctypes.c_char_p,
}

def cint(name):
    ty = getattr(ts, name)
    cname = "c_int" if ty.signed else "c_uint"
    ctypes_map[ty] = getattr(ctypes, cname + str(ty.itemsize * 8))

for name in map(tyname, universe.int_typenames):
    cint(name)

globals().update((tyname(ty.typename), cty) for ty, cty in ctypes_map.iteritems())
float_, double, longdouble = float32, float64, float128

# ______________________________________________________________________

@consing
def struct_(fields, name=None, readonly=False, packed=False):
    class Struct(ctypes.Structure):
        _fields_ = fields
        if packed:
            _pack_ = 1

    return Struct

@consing
def function(rettype, argtypes, name=None, is_vararg=False):
    assert not is_vararg
    return ctypes.CFUNCTYPE(rettype, *argtypes)

@consing
def pointer(base_type):
    if base_type in (ctypes.c_char, ctypes.c_byte):
        return string_
    return  ctypes.POINTER(base_type)

carray  = consing(lambda base_type, size: base_type * size)