import numpy as np
from numba import njit


@njit
def foo(n):
    return np.arange(10)

arr = foo(10)
print(arr)
np.testing.assert_array_equal(foo(10), np.arange(10))
