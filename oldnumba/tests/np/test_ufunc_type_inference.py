import numpy as np

import numba
from numba import *
from numba import typesystem

tup_t = typesystem.tuple_

#------------------------------------------------------------------------
# Test data
#------------------------------------------------------------------------

a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([[1, 2], [3, 4]], dtype=np.int64)

#------------------------------------------------------------------------
# Test functions
#------------------------------------------------------------------------

# ________________ unary ufuncs _______________

# ________________ binary ufuncs _______________

@autojit
def binary_ufunc(ufunc, a, b):
    return numba.typeof(ufunc(a, b))

@autojit
def binary_ufunc_dtype(ufunc, a, b, dtype):
    return numba.typeof(ufunc(a, b, dtype=dtype))

@autojit
def binary_ufunc_dtype_positional(ufunc, a, b, dtype):
    return numba.typeof(ufunc(a, b, dtype=dtype))

# ________________ binary ufunc methods _______________

@autojit
def add_reduce(a):
    return numba.typeof(np.add.reduce(a))

@autojit
def add_reduce_axis(a, axis):
    return numba.typeof(np.add.reduce(a, axis=axis))

@autojit
def accumulate(ufunc, a):
    return numba.typeof(ufunc.accumulate(a))

@autojit
def accumulate_dtype(ufunc, a, dtype):
    return numba.typeof(ufunc.accumulate(a, dtype=dtype))

@autojit
def reduceat(ufunc, a):
    return numba.typeof(ufunc.reduceat(a))

@autojit
def reduceat_dtype(ufunc, a, dtype):
    return numba.typeof(ufunc.reduceat(a, dtype=dtype))

@autojit
def outer(ufunc, a):
    return numba.typeof(ufunc.outer(a, a))

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

def equals(a, b):
    assert a == b, (a, b, type(a), type(b))

def test_binary_ufunc():
    equals(binary_ufunc(np.add, a, b), int64[:, :])
    equals(binary_ufunc(np.subtract, a, b), int64[:, :])
    equals(binary_ufunc(np.multiply, a, b), int64[:, :])
    equals(binary_ufunc(np.true_divide, a, b), int64[:, :])
    equals(binary_ufunc(np.floor_divide, a, b), int64[:, :])
    equals(binary_ufunc(np.divide, a, b), int64[:, :])

    equals(binary_ufunc(np.bitwise_and, a, b), int64[:, :])
    equals(binary_ufunc(np.bitwise_or, a, b), int64[:, :])
    equals(binary_ufunc(np.bitwise_xor, a, b), int64[:, :])
    equals(binary_ufunc(np.left_shift, a, b), int64[:, :])
    equals(binary_ufunc(np.right_shift, a, b), int64[:, :])

    equals(binary_ufunc(np.logical_and, a, b), bool_[:, :])
    equals(binary_ufunc(np.logical_or, a, b), bool_[:, :])
    equals(binary_ufunc(np.logical_xor, a, b), bool_[:, :])
    equals(binary_ufunc(np.logical_not, a, b), bool_[:, :])

    equals(binary_ufunc(np.greater, a, b), bool_[:, :])
    equals(binary_ufunc(np.greater_equal, a, b), bool_[:, :])
    equals(binary_ufunc(np.less, a, b), bool_[:, :])
    equals(binary_ufunc(np.less_equal, a, b), bool_[:, :])
    equals(binary_ufunc(np.not_equal, a, b), bool_[:, :])
    equals(binary_ufunc(np.equal, a, b), bool_[:, :])

def test_ufunc_reduce():
    equals(add_reduce(a), int32)
    equals(add_reduce_axis(b, 1), int64[:])

def test_ufunc_accumulate():
    equals(accumulate(np.add, a), int32[:])
    equals(accumulate(np.multiply, a), int32[:])

    equals(accumulate(np.bitwise_and, a), int32[:])

    equals(accumulate(np.logical_and, a), bool_[:])

    # Test with dtype
    equals(accumulate_dtype(np.add, a, np.double), double[:])

def test_ufunc_reduceat():
    equals(reduceat(np.add, a), int32[:])
    equals(reduceat(np.multiply, a), int32[:])

    equals(reduceat(np.bitwise_and, a), int32[:])

    equals(reduceat(np.logical_and, a), bool_[:])

    # Test with dtype
    equals(reduceat_dtype(np.add, a, np.double), double[:])

def test_ufunc_outer():
    equals(outer(np.add, a), int32[:, :])
    equals(outer(np.multiply, a), int32[:, :])

    equals(outer(np.bitwise_and, a), int32[:, :])

    equals(outer(np.logical_and, a), bool_[:, :])

if __name__ == "__main__":
    test_binary_ufunc()
    test_ufunc_reduce()
    test_ufunc_accumulate()
    test_ufunc_reduceat()
    test_ufunc_outer()
