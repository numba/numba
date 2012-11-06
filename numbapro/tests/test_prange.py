"""
>>> test_prange()
"""

import numbapro
import numba
from numba import *

@autojit
def test_prange():
    sum = 0.0
    for i in numba.prange(10):
        sum += i
    return sum

if __name__ == '__main__':
    test_prange()
#    import doctest
#    doctest.testmod()