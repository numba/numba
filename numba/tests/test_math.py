from numba import *

import math
import numpy as np

def a(dtype=np.double):
    return np.arange(1, 10, dtype=dtype)

def expected(a):
    return np.sum(np.log(a) * np.sqrt(a) - np.cos(a) * np.sin(a))

@autojit(backend='ast')
def numpy_math(a):
    sum = 0.0
    for i in range(a.shape[0]):
        sum += np.log(a[i]) * np.sqrt(a[i]) - np.cos(a[i]) * np.sin(a[i])
    return sum

def test_numpy_math():
    result = numpy_math(a())
    assert result == expected(a()), (result, expected(a()))

if __name__ == "__main__":
    test_numpy_math()