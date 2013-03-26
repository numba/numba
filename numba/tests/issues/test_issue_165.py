# -*- coding: utf-8 -*-
"""
Test using a dtype that is not an actual NumPy dtype (np.bool is the
builtin bool).
"""

# Thanks to Gaëtan de Menten

import numpy as np
from numba import autojit, jit, double, b1

def loop(a):
    length = len(a)
    result = np.empty(length, dtype=np.bool)
    for i in range(length):
        result[i] = a[i] > 0.1
    return result

def test_nondtype_dtype():
    loop_nb = jit(b1[:](double[:]))(loop)
    a = np.random.rand(1e6)
    assert np.allclose(loop_nb(a), loop(a))

if __name__ == '__main__':
    test_nondtype_dtype()
