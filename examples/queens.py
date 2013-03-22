# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numba as nb
from numba import *

ListInt = nb.typeof(nb.typedlist(int_)) # Define List[int] type

@autojit
def hits(x1, y1, x2, y2):
    "Check whether a queen positioned at (x1, y1) will hit a queen at position (x2, y2)"
    return x1 == x2 or y1 == y2 or abs(x1 - x2) == abs(y1 - y2)

@autojit
def hitsany(x, y, queens_x, queens_y):
    "Check whether a queen positioned at (x1, y1) will hit any other queen"
    # TODO: optimize special methods (__iter__, __getitem__)
    for i in range(queens_x.__len__()):
        if hits(x, y, queens_x.__getitem__(i), queens_y.__getitem__(i)):
            return True

    return False

@jit(bool_(int_, ListInt, ListInt))
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

@jit(object_(int_))
def solve(n):
    queens_x = nb.typedlist(int_)
    queens_y = nb.typedlist(int_)

    if _solve(n, queens_x, queens_y):
        return queens_x, queens_y
    else:
        return None, None


print(solve(8))
# %timeit solve(8)

# Comment out @jit/@autojit
# print(solve(8))
# %timeit solve(8)

