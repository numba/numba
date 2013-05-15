# -*- coding: utf-8 -*-

"""
Support for math as a postpass on LLVM IR.
"""

from __future__ import print_function, division, absolute_import

import ctypes
import collections

from numba import *
from numba.support import ctypes_support, llvm_support
from numba.type_inference.modules import mathmodule

import llvm.core

# ______________________________________________________________________

_ints = {long_.itemsize: 'l', longlong.itemsize: 'll'}
_floats = {float32: 'f', float64: '', float128: 'l'}

_int_name     = lambda name, ty: _ints.get(ty.itemsize, '') + name
_float_name   = lambda name, ty: name + _floats.get(ty)
_complex_name = lambda name, ty: 'c' + _float_name(name, ty.base_type)

def _absname(type):
    if type.is_int:
        return _int_name('abs', type)
    elif type.is_float:
        return _float_name('fabs', type)
    else:
        return _complex_name('abs', type)

def _math_suffix(name, type):
    if type.is_float:
        return _float_name(name, type)
    else:
        return _complex_name(name, type)

# ______________________________________________________________________

def from_llvmty(llvmtype):
    "LLVM type -> numba type"
    cty = llvm_support.map_llvm_to_ctypes(llvmtype)
    return ctypes_support.from_ctypes_type(cty)

# ______________________________________________________________________

types = (#int_, long_, longlong,
         float32, float64, float128,
         complex64, complex128, complex256)

def use_openlibm():
    """
    Populate a dict with runtime addressed of openlibm math functions.

    :returns: { func_name : { return_type, argtype) : func_addr } }
    """
    libm = ctypes.CDLL(ctypes.util.find_library("openlibm"))
    funcptrs = collections.defaultdict(dict)

    def add_func(name, ty):
        func = getattr(libm, name)
        p = ctypes.cast(func, ctypes.c_void_p).value
        funcptrs[mathfunc][ty, ty] = p

    for mathfunc in mathmodule.unary_libc_math_funcs:
        for ty in types:
            print(_math_suffix(mathfunc, ty))
            # add_func(_math_suffix(mathfunc, ty), ty)

    for ty in types:
        add_func(_absname(ty), ty)

    return funcptrs

# ______________________________________________________________________

def link_llvm_math_intrinsics(engine, module, library):
    """
    Add a runtime address for all global functions named numba.math.*
    """
    # find all known math intrinsics and implement them.
    for gv in module.list_globals():
        name = gv.getName()
        if name.startswith("numba.math."):
            assert not gv.getInitializer()
            assert gv.type.kind == llvm.core.TYPE_FUNCTION

            signatures = library[gv.name]
            restype = from_llvmty(gv.return_type)
            argtype = from_llvmty(gv.args[0])

            ptr = signatures[restype, argtype]
            engine.addGlobalMapping(gv, ptr)