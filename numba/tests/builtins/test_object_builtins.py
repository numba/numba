"""
>>> get_globals()
20
>>> get_locals()
Traceback (most recent call last):
    ...
NumbaError: locals() is not supported in numba functions
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

if __name__ == '__main__':
    import doctest
    doctest.testmod()