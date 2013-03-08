"""
>>> error()
Traceback (most recent call last):
    ...
NumbaError: 13:4: object of type int cannot be sliced
"""

from numba import *

@autojit(backend='ast')
def error():
    i = 10
    i[:]

if __name__ == "__main__":
    import numba
    numba.testmod()