"""
>>> a = np.arange(10, dtype=np.double)

>>> preload_arg(a)
4.0
>>> preload_local(a)
4.0
>>> preload_phi(a)
45.0
>>> preload_phi_cycle(a)
45.0
>>> preload_phi_cycle2(a)
45.0
"""

import numpy as np
from numba import *

@autojit
def preload_arg(A):
    return A[4]

@autojit
def preload_local(A):
    A = A
    return A[4]

@autojit
def preload_phi(A):
    sum = 0.0
    for i in range(10):
        sum += A[i]
        A = A

    return sum

@autojit
def preload_phi_cycle(A):
    # A_0 = A                       # <- propagated preload
    sum = 0.0
    for i in range(10):
        # A_1 = phi(A_0, A_3)       # <- preload
        sum += A[i]
        if i > 5:
            A = A # A_2             # <- propagated preload
        # A_3 = phi(A_1, A_2)       # <- propagated preload

    return sum

@autojit
def preload_phi_cycle2(A):
    # A_0 = A                       # <- propagated preload
    sum = 0.0
    for i in range(10):
        # A_1 = phi(A_0, A_3)       # <- propagated preload
        if i > 5:
            A = A # A_2             # <- propagated preload

        # A_3 = phi(A_1, A_2)       # <- preload
        sum += A[i]


    return sum

if __name__ == "__main__":
   a = np.arange(10, dtype=np.double)
   preload_arg(a)
   preload_phi(a)
    # import numba
    # numba.testing.testmod()
