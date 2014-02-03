from __future__ import print_function, division, absolute_import

import numpy as np
from numba import autojit

def test(a, b):
    r = np.empty(3, dtype=bool)
    for i in range(len(a)):
        r[i] = a[i] != b[i]
    return r
test_nb = autojit(test)

a = np.arange(3, dtype=complex)
b = np.arange(3, dtype=complex)
b[1] += 1j

assert np.array_equal(test(a, b), test_nb(a, b))
