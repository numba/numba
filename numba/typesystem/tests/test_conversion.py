# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import ctypes
import inspect
from functools import partial

# from numba import llvm_types
from numba.typesystem.itypesystem import tyname
from numba import llvm_types
from numba.typesystem import itypesystem, universe
from numba.typesystem import (numba_typesystem as ts,
                              llvm_typesystem as lts,
                              ctypes_typesystem as cts)

typenames = universe.int_typenames + universe.float_typenames + ["void"]

def convert(ts1, ts2, conversion_type, typenames):
    for typename in typenames:
        t1 = getattr(ts1, tyname(typename))
        t2 = getattr(ts2, tyname(typename))
        converted = ts1.convert(conversion_type, t1)
        assert converted == t2, (str(t1), str(converted), str(t2))

#-------------------------------------------------------------------
# Numba -> LLVM
#-------------------------------------------------------------------

llvmt = partial(ts.convert, "llvm")

def test_llvm_numeric_conversion():
    convert(ts, lts, "llvm", typenames)

def test_llvm_pointers():
    # Test pointer conversion
    for typename in typenames:
        ty = getattr(ts, tyname(typename))
        lty = getattr(lts, tyname(typename))
        assert llvmt(ts.pointer(ty)) == lts.pointer(lty)

    p = ts.pointer(ts.pointer(ts.int_))
    lp = lts.pointer(lts.pointer(lts.int_))

    # See if the conversion works
    assert llvmt(p) == lp
    # See if the consing works
    # assert llvmt(p) is lp

def test_llvm_functions():
    functype = ts.function(ts.int_, (ts.float_,))
    lfunctype = lts.function(lts.int_, (lts.float_,))
    assert llvmt(functype) == lfunctype

def test_llvm_complex():
    c1 = llvmt(ts.complex128)
    c2 = lts.struct_([('real', lts.double), ('imag', lts.double)])
    c3 = lts.struct_([('real', lts.double), ('imag', lts.double)])
    assert c1 == c2
    # assert c1 is c2
    # assert c2 is c3 # enable after upgrading llvmpy to include type hash fix

def test_llvm_object():
    assert llvmt(ts.object_) == llvm_types._pyobject_head_struct_p

def test_llvm_array():
    assert llvmt(ts.array(ts.double, 1)) == llvm_types._numpy_array
    assert llvmt(ts.array(ts.int_, 2)) == llvm_types._numpy_array
    assert llvmt(ts.array(ts.object_, 3)) == llvm_types._numpy_array

def test_llvm_range():
    assert llvmt(ts.range_) == llvm_types._pyobject_head_struct_p

#-------------------------------------------------------------------
# Numba -> ctypes
#-------------------------------------------------------------------

ct = partial(ts.convert, "ctypes")

def test_ctypes_numeric_conversion():
    convert(ts, cts, "ctypes", typenames)

def test_ctypes_pointers(): # TODO: unifiy with test_llvm_pointers
    # Test pointer conversion
    for typename in typenames:
        ty = getattr(ts, tyname(typename))
        cty = getattr(cts, tyname(typename))
        assert ct(ts.pointer(ty)) == cts.pointer(cty)

    p = ts.pointer(ts.pointer(ts.int_))
    cp = cts.pointer(cts.pointer(cts.int_))

    # See if the conversion works
    assert ct(p) == cp
    # See if the consing works
    # assert ct(p) is cp

def test_ctypes_functions(): # TODO: unifiy with test_llvm_functions
    functype = ts.function(ts.int_, (ts.float_,))
    cfunctype = cts.function(cts.int_, (cts.float_,))
    assert ct(functype) == cfunctype

def test_ctypes_complex():
    c1 = ct(ts.complex128)
    c2 = cts.struct_([('real', cts.double), ('imag', cts.double)])
    c3 = cts.struct_([('real', cts.double), ('imag', cts.double)])
    assert c1._fields_ == c2._fields_, (c1._fields_, c2._fields_)

def test_ctypes_object():
    assert ct(ts.object_) == ctypes.py_object

def test_ctypes_array():
    assert ct(ts.array(ts.double, 1)) == ctypes.py_object
    assert ct(ts.array(ts.int_, 2)) == ctypes.py_object
    assert ct(ts.array(ts.object_, 3)) == ctypes.py_object

def test_ctypes_string():
    assert ct(ts.string_) == ctypes.c_char_p
    assert ct(ts.char.pointer()) == ctypes.c_char_p

if __name__ == "__main__":
    # print(ct(ts.array(ts.double, 1)))
    for name, f in globals().items():
        if name.startswith("test_") and inspect.isfunction(f):
            f()
