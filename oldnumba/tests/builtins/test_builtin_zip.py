# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import *

@autojit
def zip1(L1, L2):
    """
    >>> zip1(range(2), range(5, 8))
    [(0, 5), (1, 6)]
    """
    return list(zip(L1, L2))

@autojit
def zip2(L1, L2, L3):
    """
    >>> zip2(range(2), range(5, 8), range(9, 13))
    [(0, 5, 9), (1, 6, 10)]
    """
    return list(zip(L1, L2, L3))

@autojit
def ziploop1(L1, L2):
    """
    >>> ziploop1(range(2), range(5, 8))
    0 5
    1 6
    """
    for i, j in zip(L1, L2):
        print(i, j)

@autojit
def ziploop2(L1, L2, L3):
    """
    >>> ziploop2(range(2), range(5, 8), range(9, 13))
    0 5 9
    1 6 10
    """
    for i, j, k in zip(L1, L2, L3):
        print(i, j, k)


if __name__ == '__main__':
    import numba
    numba.testing.testmod()
