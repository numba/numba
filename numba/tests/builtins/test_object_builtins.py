"""
>>> get_globals()
20
>>> get_locals()
Traceback (most recent call last):
    ...
NumbaError: locals() is not supported in numba functions
>>> get_sum(3)
6
>>> eval_something("'hello'")
'hello'
>>> list(enumerate_list())
[(0, 1), (1, 2), (2, 3)]
>>> max_(20) == 20
True
>>> min_(-2) == -2
True
"""

from numba import *

myglobal = 20

@autojit(backend='ast')
def get_globals():
    return globals()['myglobal']

@autojit(backend='ast')
def get_locals():
    x = 2
    return locals()['x']

@autojit(backend='ast')
def get_sum(x):
    return sum([1, 2, x])

@autojit(backend='ast')
def eval_something(s):
    return eval(s)

@autojit(backend='ast')
def enumerate_list():
    return enumerate([1, 2, 3])

@autojit(backend='ast')
def max_(x):
    return max(1, 2.0, x, 14)

@autojit(backend='ast')
def min_(x):
    return min(1, 2.0, x, 14)

if __name__ == '__main__':
    import doctest
    doctest.testmod()