from numba import *

import math
import numpy as np

def a(dtype=np.double):
    return np.arange(1, 10, dtype=dtype)

def expected(a):
    return np.sum(np.log(a) * np.sqrt(a) - np.cos(a) * np.sin(a))

def expected2(a):
    return np.sum(np.expm1(a) + np.ceil(a + 0.5) * np.rint(a + 1.5))

@autojit(backend='ast')
def numpy_math(a):
    sum = 0.0
    for i in range(a.shape[0]):
        sum += np.log(a[i]) * np.sqrt(a[i]) - np.cos(a[i]) * np.sin(a[i])
    return sum

@autojit(backend='ast')
def numpy_math2(a):
    sum = 0.0
    for i in range(a.shape[0]):
        sum += np.expm1(a[i]) + np.ceil(a[i] + 0.5) * np.rint(a[i] + 1.5)
    return sum

dtypes = np.float32, np.float64 #, np.float128

def test_numpy_math():
    for dtype in dtypes:
        print dtype

        array = a(dtype)
        result = numpy_math(array)
        assert np.allclose(result, expected(array)), (result, expected(array))

        result = numpy_math2(array)
        assert np.allclose(result, expected2(array))

@autojit(backend='ast')
def power(x, y):
    return x ** y

def test_power():
    assert power(5.0, 2.0) == 25.0
    assert power(5, 2) == 25

@autojit(backend='ast')
def modulo(x, y):
    return x % y

def test_modulo():
    assert np.allclose(modulo(22.5, 0.2), 0.1)
    assert modulo(5, 2) == 1

if __name__ == "__main__":
    test_numpy_math()
    test_power()
    test_modulo()