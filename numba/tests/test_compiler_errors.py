"""
>>> error()
Traceback (most recent call last):
    ...
NumbaError: 17:4: object of type int cannot be sliced
"""

import sys
import ctypes

from numba import *
import numpy as np

@autojit(backend='ast')
def error():
    i = 10
    i[:]

if __name__ == "__main__":
    import doctest
    doctest.testmod()