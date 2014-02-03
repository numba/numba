# -*- coding: utf-8 -*-
# The best speedup is to just decorate hits() but this is to demonstrate
# the support of untyped object.
from __future__ import print_function, division, absolute_import
from numba import *

# Support for typedlist is removed in this release
# ListInt = nb.typeof(nb.typedlist(int_)) # Define List[int] type

@jit(boolean(int32, int32, int32, int32))
def hits(x1, y1, x2, y2):
    "Check whether a queen positioned at (x1, y1) will hit a queen at position (x2, y2)"
    return x1 == x2 or y1 == y2 or abs(x1 - x2) == abs(y1 - y2)

@jit
def hitsany(x, y, queens_x, queens_y):
    "Check whether a queen positioned at (x1, y1) will hit any other queen"
    for i in range(len(queens_x)):
        if hits(x, y, queens_x[i], queens_y[i]):
            return True

    return False

@jit
def _solve(n, queens_x, queens_y):
    "Solve the queens puzzle"
    if n == 0:
        return True

    for x in range(1, 9):
        for y in range(1, 9):
            if not hitsany(x, y, queens_x, queens_y):
                queens_x.append(x)
                queens_y.append(y)

                if _solve(n - 1, queens_x, queens_y):
                    return True

                queens_x.pop()
                queens_y.pop()

    return False

@jit
def solve(n):
    queens_x = []
    queens_y = []

    if _solve(n, queens_x, queens_y):
        return queens_x, queens_y
    else:
        return None, None


solve.disable_compile()
_solve.disable_compile()
hitsany.disable_compile()

print(solve(8))
# %timeit solve(8)

# Comment out @jit/@autojit
# print(solve(8))
# %timeit solve(8)

