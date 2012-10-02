import sys

import numpy as np

from numba import *
from numba.decorators import jit, autojit
from numba.tests import test_support

a = np.arange(80).reshape(8, 10)

@autojit(backend='ast')
def np_sum(a):
    return np.sum(a, axis=0)

@autojit(backend='ast')
def np_copy(a):
    return a.copy(order='F')

@autojit(backend='ast')
def attributes(a):
    return (a.T,
            a.T.T,
            a.copy(),
            np.array(a, dtype=np.double))

def test_numpy_attrs():
    result = np_sum(a)
    np_result = np.sum(a, axis=0)
    assert np.all(result == np_result)
    assert np_copy(a).strides == a.copy(order='F').strides
    assert all(np.all(result1 == result2)
                   for result1, result2 in zip(attributes(a),
                                               attributes.py_func(a)))

if __name__ == "__main__":
    test_numpy_attrs()