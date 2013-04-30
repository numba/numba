# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

from functools import partial

from numba.typesystem import typesystem, universe
from numba.typesystem import numba_typesystem as ts, llvm_typesystem as lts

llvmt = partial(ts.convert, "llvm")

typenames = universe.int_typenames + universe.float_typenames + ["void"]

def convert(ts1, ts2, conversion_type, typenames):
    for typename in typenames:
        t1 = getattr(ts1, typename)
        t2 = getattr(ts2, typename)
        assert ts.convert(conversion_type, t1) == t2, (t1, t2)

def test_numeric_conversion():
    convert(ts, lts, "llvm", typenames)

def test_pointers():
    # Test pointer conversion
    for typename in typenames:
        ty = getattr(ts, typename)
        lty = getattr(lts, typename)
        assert llvmt(ts.pointer(ty)) == lts.pointer(lty)

    p = ts.pointer(ts.pointer(ts.int))
    lp = lts.pointer(lts.pointer(lts.int))

    # See if the conversion works
    assert llvmt(p) == lp
    # See if the consing works
    assert llvmt(p) is lp

def test_functions():
    functype = ts.function(ts.int, (ts.float,))
    lfunctype = lts.function(lts.int, (lts.float,))
    assert llvmt(functype) == lfunctype

def test_complex():
    c1 = llvmt(ts.complex128)
    c2 = lts.struct([('real', lts.double), ('imag', lts.double)])
    c3 = lts.struct([('real', lts.double), ('imag', lts.double)])
    assert c1 == c2
    # assert c1 is c2
    assert c2 is c3

if __name__ == "__main__":
    test_numeric_conversion()
    test_pointers()
    test_functions()
    test_complex()