"""
>>> prange_reduction()
45.0
"""

import numbapro
import numba
from numba import utils
from numba import *

@autojit
def prange_reduction():
    sum = 0.0
    for i in numba.prange(10):
        sum += i
    return sum

@autojit
def prange_reduction_error():
    for i in numba.prange(10):
        sum += i
    sum = 0.0
    return sum

__doc__ += """
>>> prange_reduction_error()
Traceback (most recent call last):
    ...
NumbaError: 21:8: Reduction variable 'sum' must be initialized before the loop
"""

@autojit
def prange_reduction_and_privates():
    sum = 0.0
    for i in numba.prange(10):
        j = i * 2
        sum += i

    return sum

__doc__ += """
>>> prange_reduction_and_privates()
"""

if __name__ == '__main__':
    prange_reduction_and_privates()
    import doctest
    doctest.testmod()