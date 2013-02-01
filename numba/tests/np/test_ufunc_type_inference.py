import numpy as np

import numba
from numba import *
from numba import typesystem

tup_t = typesystem.TupleType

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

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

def equals(a, b):
    assert a == b, (a, b, type(a), type(b),
                    a.comparison_type_list, b.comparison_type_list)

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

if __name__ == "__main__":
    test_binary_ufunc()
#    test_ufunc_reduce()
