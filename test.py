# from numba import *
from numba.gumath import jit_xnd
from xnd import xnd
import numpy as np
from numba import *



@jit_xnd('... * D, ... * D -> ... * D')
def add_two(a, b):
    return a + b

print(add_two(xnd([2, 1, 1]), xnd([4, 5, 6])))

print(add_two(xnd([2.3, 1.3, 1.3]), xnd([4.33, 5.3, 6.3])))


@jit_xnd('... * float64 -> ... * float64')
def sin_thing(a):
    return np.sin(a)

print(sin_thing(xnd([[2.0], [1.0], [23.0]])))


@jit_xnd('K * N * D, D -> K * N * D')
def g(x, y, res):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            res[i][j] = x[i][j] + y


print(g(xnd([[2, 3, 4], [5, 6, 7]]), xnd(4)))


# using notation from https://en.wikipedia.org/wiki/Matrix_multiplication#Definition
@jit_xnd('N * M * D, M * P * D -> N * P * D',)
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

print(matrix_multiply(
    xnd([[-2, 5], [1, 6], [-4, -1]]),
    xnd([[2, 7], [8, -3]]) 
))


print(matrix_multiply(
    xnd([[123.023]]),
    xnd([[23.2323]]) 
))
