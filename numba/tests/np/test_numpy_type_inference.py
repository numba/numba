import numpy as np

import numba
from numba import *
from numba import typesystem

tup_t = typesystem.TupleType

#------------------------------------------------------------------------
# Test functions
#------------------------------------------------------------------------

@autojit
def array(value):
    return numba.typeof(np.array(value))

@autojit
def nonzero(value):
    return numba.typeof(np.nonzero(value))

@autojit
def where(value):
    return numba.typeof(np.where(value))

@autojit
def where3(value, x, y):
    return numba.typeof(np.where(value, x, y))

@autojit
def numba_dot(A, B):
    result = np.dot(A, B)
    return numba.typeof(result), result

@autojit
def sum(a):
    return numba.typeof(np.sum(a))

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

def equals(a, b):
    assert a == b, (a, b, a.comparison_type_list, b.comparison_type_list)

def test_array():
    equals(array(np.array([1, 2, 3], dtype=np.double)), float64[:])
    equals(array(np.array([[1, 2, 3]], dtype=np.int32)), int32[:, :])
    equals(array(np.array([[1, 2, 3],
                           [4, 5, 6]], dtype=np.int32).T), int32[:, :])

def test_nonzero():
    equals(nonzero(np.array([1, 2, 3], dtype=np.double)),
           tup_t(npy_intp[:], 1))
    equals(nonzero(np.array([[1, 2, 3]], dtype=np.double)),
           tup_t(npy_intp[:], 2))
    equals(nonzero(np.array((((1, 2, 3),),), dtype=np.double)),
           tup_t(npy_intp[:], 3))

def test_where():
    equals(where(np.array([1, 2, 3], dtype=np.double)),
           tup_t(npy_intp[:], 1))

    equals(where3(np.array([True, False, True]),
                  np.array([1, 2, 3], dtype=np.double),
                  np.array([1, 2, 3], dtype=np.complex128)),
           complex128[:])

    equals(where3(np.array([True, False, True]),
                  np.array([1, 2, 3], dtype=np.float32),
                  np.array([1, 2, 3], dtype=np.int64)),
           float64[:])

def test_numba_dot():
    A = np.array(1)
    B = np.array(2)

    for i in range(1, 10):
        for j in range(1, 10):
            shape_A = (1,) * i
            shape_B = (1,) * j

            x = A.reshape(*shape_A)
            y = B.reshape(*shape_B)

            result_type, result = numba_dot(x, y)

            assert result == np.dot(x, y)
            assert result.ndim == result_type.ndim
            # assert result.dtype == result_type.get_dtype()

if __name__ == "__main__":
    test_array()
    test_nonzero()
    test_where()
    test_numba_dot()