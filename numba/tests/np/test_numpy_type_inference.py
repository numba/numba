import numpy as np

import numba
from numba import *
from numba import typesystem
from numba.typesystem import tuple_

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
def numba_vdot(A, B):
    result = np.vdot(A, B)
    return numba.typeof(result), result

@autojit
def numba_inner(a, b):
    result = np.inner(a, b)
    return numba.typeof(result), result

@autojit
def numba_outer(a, b):
    result = np.outer(a, b)
    return numba.typeof(result), result

@autojit
def numba_tensordot(a, b, axes):
    result = np.tensordot(a, b, axes)
    return numba.typeof(result), result

@autojit
def numba_tensordot2(a, b):
    result = np.tensordot(a, b)
    return numba.typeof(result), result

@autojit
def numba_kron(a, b):
    result = np.kron(a, b)
    return numba.typeof(result), result

# ------------- Test sum ------------

@autojit
def sum_(a):
    return numba.typeof(np.sum(a))

@autojit
def sum_axis(a, axis):
    return numba.typeof(np.sum(a, axis=axis))

@autojit
def sum_dtype(a, dtype):
    return numba.typeof(np.sum(a, dtype=dtype))

@autojit
def sum_out(a, out):
    return numba.typeof(np.sum(a, out=out))

# ------------- Basic tests ------------

@autojit
def array_from_list():
    ids = np.array([3,4,5])
    ids2 = ids < 4
    return ids2

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

def equals(a, b):
    assert a == b, (a, b, type(a), type(b))

def test_array():
    equals(array(np.array([1, 2, 3], dtype=np.double)), float64[:])
    equals(array(np.array([[1, 2, 3]], dtype=np.int32)), int32[:, :])
    equals(array(np.array([[1, 2, 3],
                           [4, 5, 6]], dtype=np.int32).T), int32[:, :])

def test_nonzero():
    equals(nonzero(np.array([1, 2, 3], dtype=np.double)),
           tuple_(npy_intp[:], 1))
    equals(nonzero(np.array([[1, 2, 3]], dtype=np.double)),
           tuple_(npy_intp[:], 2))
    equals(nonzero(np.array((((1, 2, 3),),), dtype=np.double)),
           tuple_(npy_intp[:], 3))

def test_where():
    equals(where(np.array([1, 2, 3], dtype=np.double)),
           tuple_(npy_intp[:], 1))

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

    dtype = typesystem.from_numpy_dtype(A.dtype).dtype

    for i in range(1, 5):
        for j in range(1, 5):
            # print i, j

            shape_A = (1,) * i
            shape_B = (1,) * j

            x = A.reshape(*shape_A)
            y = B.reshape(*shape_B)

            result_type, result = numba_dot(x, y)

            assert result == np.dot(x, y)
            if i + j - 2 > 0:
                assert result.ndim == result_type.ndim
            else:
                assert result_type == dtype, (result_type, dtype)

def test_numba_vdot():
    for a, b in ((np.array([1+2j,3+4j]),
                  np.array([5+6j,7+8j])),
                 (np.array([[1, 4], [5, 6]]),
                  np.array([[4, 1], [2, 2]]))):
        result_type, result = numba_vdot(a, b)
        assert result == np.vdot(a, b)
        assert result_type == typesystem.from_numpy_dtype(a.dtype).dtype
        result_type, result = numba_vdot(b, a)
        assert result == np.vdot(b, a)
        assert result_type == typesystem.from_numpy_dtype(b.dtype).dtype

def test_numba_inner():
    # Note these tests assume that the lhs' type is the same as the
    # promotion type for both arguments.  They will fail if additional
    # test data doesn't adhere to this policy.
    for a, b in ((np.array([1,2,3]), np.array([0,1,0])),
                 (np.arange(24).reshape((2,3,4)), np.arange(4)),
                 (np.eye(2), 7)):
        result_type, result = numba_inner(a, b)
        if result_type.is_array:
            assert (result == np.inner(a, b)).all()
            assert (result_type.dtype ==
                    typesystem.from_numpy_dtype(result.dtype).dtype)
            assert (result_type.dtype ==
                    typesystem.from_numpy_dtype(a.dtype).dtype)
        else:
            assert result == np.inner(a, b)
            assert result_type == typesystem.from_numpy_dtype(a.dtype).dtype

def test_numba_outer():
    for a, b in ((np.ones((5,)), np.linspace(-2, 2, 5)),
                 (1j * np.linspace(2, -2, 5), np.ones((5,))),
                 (np.array(['a', 'b', 'c'], dtype=object), np.arange(1,4)),
                 (np.array([1]), 1),
                 (np.ones((2,2,2)), np.linspace(-2, 2, 5))):
        result_type, result = numba_outer(a, b)
        assert (result == np.outer(a, b)).all()
        assert (result_type.is_array and result_type.ndim == 2)
        assert result_type.dtype == typesystem.from_numpy_dtype(a.dtype).dtype

def test_numba_tensordot():
    for a, b, axes in ((np.arange(60.).reshape(3, 4, 5),
                        np.arange(24.).reshape(4, 3, 2), ([1,0],[0,1])),
                       ):
        result_type, result = numba_tensordot(a, b, axes)
        assert (result == np.tensordot(a, b, axes)).all()
        # See comments in the docstring for
        # numba.type_inference.modules.numpymodule.tensordot().
        assert result_type == object_

def test_numba_tensordot2():
    A = np.array(1)
    B = np.array(2)

    dtype = typesystem.from_numpy_dtype(A.dtype).dtype
    for i in range(2, 5):
        for j in range(2, 5):

            shape_A = (1,) * i
            shape_B = (1,) * j

            x = A.reshape(*shape_A)
            y = B.reshape(*shape_B)

            result_type, result = numba_tensordot2(x, y)
            control = np.tensordot(x, y)
            assert result == control
            #assert result_type == numba.typeof(control)
            if i + j - 4 > 0:
                assert result.ndim == result_type.ndim
            else:
                assert result_type == dtype

def test_sum():
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([[1, 2], [3, 4]], dtype=np.int64)

    equals(sum_(a), int32)
    equals(sum_axis(a, 0), int32)
    equals(sum_dtype(a, np.double), double)
    equals(sum_out(b, a), int32[:]) # Not a valid call to sum :)

def test_basic():
    assert np.all(array_from_list() == np.array([True, False, False]))

if __name__ == "__main__":
    test_array()
    test_nonzero()
    test_where()
    test_numba_dot()
    test_numba_vdot()
    test_numba_inner()
    test_numba_outer()
    test_numba_tensordot()
    test_numba_tensordot2()
    test_sum()
    test_basic()
