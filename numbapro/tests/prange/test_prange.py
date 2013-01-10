
"""
>>> prange_reduction()
45.0
>>> prange_reduction2()
49999995000000.0
"""

import numbapro
import numba
from numba import utils
from numba import *

import numpy as np

@autojit
def prange_reduction():
    sum = 0.0
    for i in numba.prange(10):
        sum += i
    return sum

@autojit
def prange_reduction2():
    sum = 0.0
    for i in numba.prange(10000000):
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
NumbaError: 32:8: Local variable  'sum' is not bound yet
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
def prange_lastprivate():
    sum = 10.0
    for i in numba.prange(10):
        j = i * 2
        sum += j

    print j
    return sum

__doc__ += """
>>> prange_lastprivate()
18
100.0
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

@autojit
def test_sum2d(A):
    sum = 0.0
    for i in numba.prange(A.shape[0]):
        for j in range(A.shape[1]):
            # print i, j
            sum += A[i, j]

    return sum

__doc__ += """
>>> a = np.arange(100).reshape(10, 10)
>>> test_sum2d(a)
4950.0
>>> test_sum2d(a.astype(np.complex128))
(4950+0j)
>>> np.sum(a)
4950
"""

@autojit
def test_prange_in_closure(x):
    sum = 10.0
    N = 10

    @double()
    def inner():
        sum = 100.0
        for i in numba.prange(N):
            for j in range(N):
                sum += i * x

        return sum

    return inner

__doc__ += """
>>> test_prange_in_closure(2.0)()
1000.0
"""

@autojit
def test_prange_in_closure2(x):
    sum = 10.0
    N = 10

    @double()
    def inner():
        sum = 100.0
        for i in numba.prange(N):
            for j in range(N):
                sum += (i * N + j) * x

        return sum

    return inner

__doc__ += """
>>> test_prange_in_closure2(2.0)()
10000.0
"""

if __name__ == '__main__':
    jit(double(double))(prange_shared_privates_reductions.py_func)
    jit(double(double[:, :]))(test_sum2d.py_func)

#    print prange_shared_privates_reductions(2.0)
#    a = np.arange(100).reshape(10, 10)
#    print test_sum2d(a)

#    print test_prange_in_closure(2.0)()
#    print test_prange_in_closure2(2.0)()
#    print test_prange_in_closure2(2)()

#    import doctest
#    doctest.testmod()