# -*- coding: utf-8 -*-

"""
>>> f(1, 2)
(2, 1)
"""

from __future__ import print_function, division, absolute_import

import numba
from numba.testing import testmod

@numba.autojit
def f(a, b):
    for i in range(1):
        b, a = a, b

    return a, b

testmod()
