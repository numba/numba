# -*- coding: utf-8 -*-

"""
>>> unpack_loop()
Traceback (most recent call last):
      ...
NumbaError: ...: Only a single target iteration variable is supported at the moment
"""

from __future__ import print_function, division, absolute_import

import numba
from numba.testing import testmod

@numba.autojit(warnstyle="simple")
def unpack_loop():
    x = [(1,2),(3,4)]
    for (a, b) in x:
        print(a + b)

testmod()
