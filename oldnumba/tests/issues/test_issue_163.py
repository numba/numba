# -*- coding: utf-8 -*-
# Thanks to Gaëtan de Menten

"""
>>> test_valid_compare()
>>> invalid_compare(np.arange(10))
Traceback (most recent call last):
      ...
NumbaError: 27:11: Cannot determine truth value of boolean array (use any or all)
"""

import numpy as np
from numba import autojit, jit, double, b1

def array(a):
    return a > 0.1

#fails too
#array_nb = jit(b1[:](double[:]))(array)
def test_valid_compare():
    array_nb = autojit(array)
    a = np.random.rand(1e6)
    assert np.allclose(array(a), array_nb(a))

@autojit(warnstyle="simple")
def invalid_compare(a):
    return 1 < a < 2


if __name__ == '__main__':
    from numba.testing import test_support
    test_support.testmod()
