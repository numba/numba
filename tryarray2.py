import numpy as np
from numba import njit


@njit
def bar(x):
    return x + x

@njit
def foo(n):
    arr = np.arange(n, dtype=np.float32)
    for i in range(n):
        arr = arr + (bar(arr)/(i + 2))
    return arr


arr = foo(10)
print(arr)
np.testing.assert_array_equal(foo(10), foo.py_func(10))

