from numba import *
from numba.gumath import jit_xnd
from xnd import xnd
import numpy as np

# @guvectorize([(float64[:, :], float64, float64[:, :])], '(n, n),()->(n, n)')
# def g(x, y, res):
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             res[i][j] = x[i][j] + y

@jit_xnd(
    '... * int64, ... * int64 -> ... * int64',
    'int64(int64, int64)',
    [0, 0, 0]
)
def add_two(a, b):
    return a + b

print(add_two(xnd([2, 1, 1]), xnd([4, 5, 6])))


@jit_xnd('... * float64 -> ... * float64', 'float64(float64)', [0, 0])
def sin_thing(a):
    return np.sin(a)

print(sin_thing(xnd([[2.0], [1.0], [23.0]])))


@jit_xnd(
    'K * N * int64, int64 -> K * N * int64',
    '(int64[:, :], int64, int64[:, :])',
    [2, 0, 2],
)
def g(x, y, res):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            res[i][j] = x[i][j] + y


print(g(xnd([[2, 3, 4], [5, 6, 7]]), xnd(4)))


# using notation from https://en.wikipedia.org/wiki/Matrix_multiplication#Definition
@jit_xnd(
    'N * M * int64, N * P * int64 -> N * P * int64',
    '(int64[:, :], int64[:, :], int64[:, :])',
    [2, 2, 2],
)
def matrix_multiply(a, b, c):
    n, m = a.shape
    m, p = b.shape
    for i in range(n):
        for j in range(p):
            c[i][j] = 0
            for k in range(m):
                c[i][j] += a[i][k] * b[k][j]


print(matrix_multiply(
    xnd([[0, 1], [0, 0]]),
    xnd([[0, 0], [1, 0]]) 
))
