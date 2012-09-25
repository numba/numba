import numbapro
from numba.decorators import function

import numpy as np

@function
def array_expr(a, b, c):
    return a + b * c

@function
def func(a):
    return a * 2.0

@function
def array_expr2(a, b, c):
    return a + b + func(c)

@function
def array_expr3(a, b, c):
    a[...] = a + b * c

def test_array_expressions():
    a = np.arange(120).reshape(10, 12).astype(np.float32)
    assert np.all(array_expr(a, a, a) == array_expr.py_func(a, a, a))
    assert np.all(array_expr2(a, a, a) == array_expr2.py_func(a, a, a))

    result, numpy_result = a.copy(), a.copy()
    array_expr3(result, result, result)
    array_expr3.py_func(numpy_result, numpy_result, numpy_result)
    assert np.all(result == numpy_result)

@function
def array_expr_matmul(A, B):
    m, n = A.shape
    n, p = B.shape
    C = np.empty((m, p), dtype=A.dtype)
    for i in range(m):
        for j in range(p):
            C[i, j] = (A[i, :] * B[:, j]).sum()

    return C

def test_matmul():
    a = np.arange(120).reshape(10, 12).astype(np.float32)
    b = a.T
    result = array_expr_matmul(a, b)
    assert np.all(result == np.dot(a, b))

if __name__ == '__main__':
    test_array_expressions()
    test_matmul()