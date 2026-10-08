import numpy as np
from numba import njit


@njit(cache=True)
def bar(x):
    return x + x


res = bar(5)
assert res == bar.py_func(5)
print(res)

@njit(cache=True)
def foo(n):
    arr = np.arange(n, dtype=np.float32)
    for i in range(n):
        arr = arr + (bar(arr)/(i + 2))
    return arr


arr = foo(10)
print(arr)
np.testing.assert_array_equal(foo(10), foo.py_func(10))

