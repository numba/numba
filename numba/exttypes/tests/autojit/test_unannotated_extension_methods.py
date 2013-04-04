from itertools import cycle

from numba import *
from numba.testing.test_support import parametrize, main

import numpy as np

# NOTE: We make these two separate classes because we don't want a set of
# NOTE: precompiled methods to affect the tests

del python, nopython # Make sure we run in *compiled* mode

def _make_list_func(self, A):
    L = []

    with nopython:
        for i in range(A.shape[0]):
            item = A[i]

            with python:
                L.append(item)

    return L

def make_list_func(self, A):
    return self._make_list(A)

@autojit
class Base1(object):
    """
    Test numba calling autojit methods
    """
    def __init__(self, myfloat):
        self.value = myfloat

    def getvalue(self):
        return self.value

    _make_list = _make_list_func
    make_list = make_list_func

@autojit
class Base2(object):
    """
    Test numba calling autojit methods
    """
    def __init__(self, myfloat):
        self.value = myfloat

    def getvalue(self):
        return self.value

    _make_list = _make_list_func
    make_list = make_list_func

@autojit
class Base3(object):
    """
    Test Python calling autojit methods.
    """

    def __init__(self, myfloat):
        self.value = myfloat

    def getvalue(self):
        return self.value

    make_list = _make_list_func

@autojit
def run(obj, array):
    list = obj.make_list(array)
    return list

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

obj1 = Base1(10.0)
obj2 = Base2(10.0)
obj3 = Base3(10.0)

assert obj1.value == obj2.value == obj3.value == 10.0

dtypes = (
    np.float32, np.int32,
    np.double, np.int64,
    np.complex64, np.complex128,
)

params = (list(zip(cycle([obj1]), dtypes)) +
          list(zip(cycle([obj3]), dtypes)))

# ______________________________________________________________________
# Parameterized tests

@parametrize(*params)
def test_python_specialize_method(param):
    obj, dtype = param

    A = np.arange(10, dtype=dtype)
    L = obj.make_list(A)
    assert np.all(A == L)

@parametrize(*zip(cycle([obj2]), dtypes))
def test_numba_func_use_method(param):
    obj, dtype = param

    A = np.arange(10, dtype=dtype)
    L = run(obj, A)
    assert np.all(A == L)


if __name__ == '__main__':
    # obj, dtype = obj2, np.double
    #
    # A = np.arange(10, dtype=dtype)
    # L = run(obj, A)
    # assert np.all(A == L)

    main()
