from itertools import cycle

from numba import *
from numba.testing.test_support import parametrize, main

import numpy as np

# NOTE: We make these two separate classes because we don't want a set of
# NOTE: precompiled methods to affect the tests

del python, nopython # Make sure we run in *compiled* mode

@autojit
class Base1(object):
    """
    Test numba calling autojit methods
    """
    def __init__(self, myfloat):
        self.value = myfloat

    def getvalue(self):
        return self.value

    def _make_list(self, A):
        L = []

        with nopython:
            for i in range(A.shape[0]):
                item = A[i]

                with python:
                    L.append(item)

        return L

    def make_list(self, A):
        return self._make_list(A)

@autojit
class Base2(object):
    """
    Test Python calling autojit methods.
    """

    def __init__(self, myfloat):
        self.value = myfloat

    def getvalue(self):
        return self.value

    def make_list(self, A):
        L = []

        with nopython:
            for i in range(A.shape[0]):
                item = A[i]

                with python:
                    L.append(item)

        return L

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

obj1 = Base1(10.0)
obj2 = Base2(10.0)

dtypes = (
    np.float32, np.int32,
    np.double, np.int64,
    np.complex64, np.complex128,
)

params = (# zip(cycle([obj1]), dtypes) +
          zip(cycle([obj2]), dtypes))

@parametrize(*params)
def test_python_specialize_method(param):
    obj, dtype = param

    A = np.arange(10, dtype=dtype)
    L = obj.make_list(A)
    assert np.all(A == L)

if __name__ == '__main__':
    main()