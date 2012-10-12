"""
>>> empty_float()
0.0
>>> float_conjugate()
1.5
"""

import sys

from numba import *

@autojit(backend='ast')
def empty_float(y):
    x = float(y)
    return x

@autojit(backend='ast')
def float_conjugate():
    return 1.5.conjugate()

if __name__ == '__main__':
    import doctest
    doctest.testmod()