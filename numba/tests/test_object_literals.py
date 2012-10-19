"""
>>> get_list('world')
[1, 'hello', 2.0, 'world']
>>> get_tuple('world')
(1L, 'hello', 2.0, 'world')
>>> get_dict('world') == {"hello": 1, 2.0: 'world'}
True
"""

import sys
from numba import *

myglobal = 20

@autojit(backend='ast')
def get_list(x):
    return [1L, "hello", 2.0, x]

@autojit(backend='ast')
def get_tuple(x):
    return (1L, "hello", 2.0, x)

@autojit(backend='ast')
def get_dict(x):
    return {"hello": 1L, 2.0: x}

if __name__ == '__main__':
    import doctest
    doctest.testmod()