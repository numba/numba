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
    sum = 10.0
    for i in numba.prange(10):
        j = i * 2
        sum += j

    return sum

__doc__ += """
>>> prange_reduction_and_privates()
100.0
"""

@autojit
def prange_privates_error():
    sum = 10.0
    for i in numba.prange(10):
        j = i * 2
        sum += j

    print j
    return sum

__doc__ += """
>>> prange_privates_error()
Traceback (most recent call last):
    ...
NumbaError: 53:10: Local variable  'j' is not bound yet
"""

@autojit
def prange_shared_privates_reductions(shared):
    sum = 10.0

    for i in numba.prange(10):
        j = i * shared
        sum += j

    shared = 3.0
    return sum

__doc__ += """
>>> prange_shared_privates_reductions(2.0)
100.0
"""

if __name__ == '__main__':
#    prange_privates_error()
    import doctest
    doctest.testmod()