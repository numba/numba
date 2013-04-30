# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

from functools import partial

from numba.typesystem import typesystem, universe
from numba.typesystem import numba_typesystem as ts, llvm_typesystem as lts

def test_pointers():
    assert ts.pointer(ts.int) is ts.pointer(ts.int)

def test_functions():
    functype1 = ts.function(ts.int, (ts.float,))
    functype2 = ts.function(ts.int, (ts.float,))
    functype3 = ts.function(ts.int, (ts.float,), is_vararg=False)
    functype4 = ts.function(ts.int, (ts.float,), name="hello")
    functype5 = ts.function(ts.int, (ts.float,), name="hello", is_vararg=False)
    functype6 = ts.function(ts.int, (ts.float,), name="hello", is_vararg=True)

    assert functype1 is functype2
    assert functype1 is functype3
    assert functype1 is not functype4
    assert functype1 is not functype5
    assert functype1 is not functype6

    assert functype4 is functype5
    assert functype4 is not functype6

if __name__ == "__main__":
    test_pointers()
    test_functions()