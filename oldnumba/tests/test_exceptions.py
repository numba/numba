"""
>>> boom()
Traceback (most recent call last):
    ...
ValueError: invalid literal for int() with base 10: 'boom'
>>> boom2()
Traceback (most recent call last):
    ...
TypeError: 'object' object is not callable
>>> boom3()
Traceback (most recent call last):
    ...
TypeError: 'object' object is not callable
"""

import sys
import ctypes

from numba import *
import numpy as np

@autojit(backend='ast')
def boom():
    return int('boom')

@jit(int_())
def boom2():
    return object()('boom')

@jit(complex128())
def boom3():
    return object()('boom')

if __name__ == "__main__":
    import numba
    numba.testing.testmod()
