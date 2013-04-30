# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

from numba.typesystem import numba_typesystem as ts

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

def test_struct():
    s1 = ts.struct([('a', ts.int), ('b', ts.float)])
    s2 = ts.struct([('a', ts.int), ('b', ts.float)])
    assert s1 is not s2

def test_arrays():
    A = ts.array(ts.double, 1)
    B = ts.array(ts.double, 1)
    C = ts.array(ts.float, 1)
    D = ts.array(ts.double, 2)

    assert A is B
    assert A is not C
    assert A is not D

if __name__ == "__main__":
    test_pointers()
    test_functions()
    test_struct()
    test_arrays()