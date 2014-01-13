# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

from numba.typesystem import numba_typesystem as ts

# ______________________________________________________________________
def test_mutability():
    ty = ts.function(ts.int_, (ts.float_,))
    ty.args
    try:
        ty.args = [1, 2]
    except AttributeError as e:
        pass
    else:
        raise Exception(
            "Expected: AttributeError: Cannot set attribute 'args' of type ...")
# ______________________________________________________________________

def test_pointers():
    assert ts.pointer(ts.int_) is ts.pointer(ts.int_)

def test_functions():
    functype1 = ts.function(ts.int_, (ts.float_,))
    functype2 = ts.function(ts.int_, (ts.float_,))
    functype3 = ts.function(ts.int_, (ts.float_,), is_vararg=False)
    functype4 = ts.function(ts.int_, (ts.float_,), name="hello")
    functype5 = ts.function(ts.int_, (ts.float_,), name="hello", is_vararg=False)
    functype6 = ts.function(ts.int_, (ts.float_,), name="hello", is_vararg=True)

    assert functype1 is functype2
    assert functype1 is functype3
    assert functype1 is not functype4
    assert functype1 is not functype5
    assert functype1 is not functype6

    assert functype4 is functype5
    assert functype4 is not functype6

# def test_struct():
#     s1 = ts.struct_([('a', ts.int_), ('b', ts.float_)])
#     s2 = ts.struct_([('a', ts.int_), ('b', ts.float_)])
#     assert s1 is not s2

def test_arrays():
    A = ts.array(ts.double, 1)
    B = ts.array(ts.double, 1)
    C = ts.array(ts.float_, 1)
    D = ts.array(ts.double, 2)

    assert A is B
    assert A is not C
    assert A is not D

def test_complex():
    assert ts.complex_(ts.float_) is ts.complex64
    assert ts.complex_(ts.double) is ts.complex128
    # assert ts.complex_(ts.longdouble) is ts.complex256

if __name__ == "__main__":
    test_mutability()
    test_pointers()
    test_functions()
    # test_struct()
    test_arrays()
    test_complex()
