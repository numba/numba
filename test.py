# from numba import *
# @guvectorize([(int64[:, :], int64, int64[:, :])], '(n, n),()->(n, n)')
# def g(x, y, res):
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             res[i][j] = x[i][j] + y

from numba.gumath import jit_to_kernel
from xnd import xnd

@jit_to_kernel('... * int64, ... * int64 -> ... * int64', 'int64(int64, int64)')
def add_two(a, b):
    return a + b

add_two(xnd([1]), xnd([4, 5, 6]))
