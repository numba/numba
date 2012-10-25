"""
>>> round_val(2.2)
2.0
>>> round_val(3.6)
4.0
>>> round_val(5)
5.0

>>> round2(10.497, 2)
10.5
>>> round2(497, -1)
500.0
"""

import numpy as np
from numba import *

@autojit(backend='ast')
def round_val(a):
    return round(a)

@autojit(backend='ast')
def round2(a, b):
    return round(a, b)

if __name__ == '__main__':
    round2(10.497, 2)
#    import doctest
#    doctest.testmod()
