"""
>>> boom()
Traceback (most recent call last):
    ...
ValueError: invalid literal for int() with base 10: 'boom'
"""

import sys
import ctypes

from numba import *
import numpy as np

def boom():
    return int('boom')

if __name__ == "__main__":
    import doctest
    doctest.testmod()