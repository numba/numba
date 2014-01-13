# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

# Based on cython/tests/run/enumerate_T316.pyx

from numba import *

@autojit
def go_py_enumerate():
    """
    >>> go_py_enumerate()
    0 1
    1 2
    2 3
    3 4
    """
    for i,k in enumerate(range(1,5)):
        print(i, k)

@autojit
def py_enumerate_list_index_target():
    """
    >>> py_enumerate_list_index_target()
    [0] 1
    [1] 2
    [2] 3
    [3] 4
    """
    target = [None]
    for target[0],k in enumerate(range(1,5)):
        print(target, k)

@autojit
def go_py_enumerate_start():
    """
    >>> go_py_enumerate_start()
    5 1
    6 2
    7 3
    8 4
    """
    for i,k in enumerate(list(range(1,5)), 5):
        print(i, k)

@autojit
def go_c_enumerate():
    """
    >>> go_c_enumerate()
    0 1
    1 2
    2 3
    3 4
    """
    for i,k in enumerate(range(1,5)):
        print(i, k)

@autojit
def go_c_enumerate_step():
    """
    >>> go_c_enumerate_step()
    0 1
    1 3
    2 5
    """
    for i,k in enumerate(range(1,7,2)):
        print(i, k)

# @autojit # TODO:
def py_enumerate_dict(d):
    """
    >>> py_enumerate_dict({})
    :: 55 99
    >>> py_enumerate_dict(dict(a=1, b=2, c=3))
    0 True
    1 True
    2 True
    :: 2 True
    """
    i = 55
    k = 99
    keys = list(d.keys())
    for i,k in enumerate(d):
        k = keys[i] == k
        print(i, k)
    print("::", i, k)

@autojit
def py_enumerate_break(t):
    """
    >>> py_enumerate_break([1,2,3,4])
    0 1
    :: 0 1
    """
    i,k = 55,99
    for i,k in enumerate(t):
        print(i, k)
        break
    print("::", i, k)

@autojit
def py_enumerate_return(t):
    """
    >>> py_enumerate_return([])
    :: 55 99
    >>> py_enumerate_return([1,2,3,4])
    0 1
    """
    i,k = 55,99
    for i,k in enumerate(t):
        print(i, k)
        return
    print("::", i, k)

@autojit
def py_enumerate_continue(t):
    """
    >>> py_enumerate_continue([1,2,3,4])
    0 1
    1 2
    2 3
    3 4
    :: 3 4
    """
    i,k = 55,99
    for i,k in enumerate(t):
        print(i, k)
        continue
    print("::", i, k)

@autojit
def empty_c_enumerate():
    """
    >>> empty_c_enumerate()
    (55, 99)
    """
    i,k = 55,99
    for i,k in enumerate(range(0)):
        print(i, k)
    return i, k

# Not supported (yet)
# @autojit
# def single_target_enumerate():
#     """
#     >>> single_target_enumerate()
#     0 1
#     1 2
#     2 3
#     3 4
#     """
#     for t in enumerate(range(1,5)):
#         print(t[0], t[1])

# @autojit # TODO:
def multi_enumerate():
    """
    >>> multi_enumerate()
    0 0 0 1
    1 1 1 2
    2 2 2 3
    3 3 3 4
    """
    for a,(b,(c,d)) in enumerate(enumerate(enumerate(range(1,5)))):
        print(a,b,c,d)

# @autojit # TODO:
def multi_enumerate_start():
    """
    >>> multi_enumerate_start()
    0 2 0 1
    1 3 1 2
    2 4 2 3
    3 5 3 4
    """
    for a,(b,(c,d)) in enumerate(enumerate(enumerate(range(1,5)), 2)):
        print(a,b,c,d)

# @autojit # TODO:
def multi_c_enumerate():
    """
    >>> multi_c_enumerate()
    0 0 0 1
    1 1 1 2
    2 2 2 3
    3 3 3 4
    """
    for a,(b,(c,d)) in enumerate(enumerate(enumerate(range(1,5)))):
        print(a,b,c,d)

@autojit
def convert_target_enumerate(L):
    """
    >>> convert_target_enumerate([2,3,5])
    0 2
    1 3
    2 5
    """
    for a, b in enumerate(L):
        print(a,b)

@autojit
def convert_target_enumerate_start(L, n):
    """
    >>> convert_target_enumerate_start([2,3,5], 3)
    3 2
    4 3
    5 5
    """
    for a, b in enumerate(L, n):
        print(a,b)


if __name__ == '__main__':
    import numba
    numba.testing.testmod()
