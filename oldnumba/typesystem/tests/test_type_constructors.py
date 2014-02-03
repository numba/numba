# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba.typesystem import numba_typesystem as ts

def test_pointers():
    p = ts.pointer(ts.pointer(ts.int_))
    assert str(p) == "int **", str(p)

def test_arrays():
    A = ts.array(ts.double, 1)
    B = ts.array(ts.double, 2)

    assert str(A) == "float64[:]"
    assert str(A[1:]) == "float64"
    assert str(B[1:]) == "float64[:]"
    assert str(B[-1:10]) == "float64[:]"
    assert str(B[0:]) == "float64[:, :]"
    assert str(B[0:10]) == "float64[:, :]"
    assert str(B[-2:10]) == "float64[:, :]"

def test_functions():
    functype = ts.function(ts.int_, (ts.float_,))
    assert str(functype) == "int (*)(float32)", functype
    functype = ts.function(ts.int_, (ts.float_,), "hello")
    assert str(functype) == "int (*hello)(float32)", functype

if __name__ == "__main__":
    test_pointers()
    test_arrays()
    test_functions()
