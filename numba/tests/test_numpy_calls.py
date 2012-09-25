import sys

import numpy as np

from numba import *
from numba.decorators import jit, function
from numba.tests import test_support

a = np.arange(80).reshape(8, 10)

@function
def np_sum(a):
    return np.sum(a, axis=0)

@function
def np_copy(a):
    return a.copy(order='F')

def test_numpy_attrs():
    result = np_sum(a)
    np_result = np.sum(a, axis=0)
    assert np.all(result == np_result)

    assert np_copy(a).strides == a.copy(order='F').strides

if __name__ == "__main__":
    test_numpy_attrs()