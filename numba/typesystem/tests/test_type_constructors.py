# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba.typesystem import numba_typesystem as ts

def test_arrays():
    A = ts.array(ts.double, 1)
    B = ts.array(ts.double, 2)

    assert str(A) == "double[:]"
    assert str(A[1:]) == "double"
    assert str(B[1:]) == "double[:]"
    assert str(B[-1:10]) == "double[:]"
    assert str(B[0:]) == "double[:, :]"
    assert str(B[0:10]) == "double[:, :]"
    assert str(B[-2:10]) == "double[:, :]"

def test_functions():
    functype = ts.function(ts.int, (ts.float,))
    assert str(functype) == "int (*)(float)", functype
    functype = ts.function(ts.int, (ts.float,), "hello")
    assert str(functype) == "int (*hello)(float)", functype

if __name__ == "__main__":
    test_arrays()
    test_functions()