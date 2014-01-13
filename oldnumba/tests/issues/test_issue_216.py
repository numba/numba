# -*- coding: utf-8 -*-

"""
>>> with_stat("foo", "bar")
Traceback (most recent call last):
      ...
NumbaError: ...: Only 'with python' and 'with nopython' is supported at this moment
"""

from __future__ import print_function, division, absolute_import

import numba
from numba.testing import testmod

@numba.autojit(warnstyle="simple")
def with_stat(fn, msg):
    with open(fn, 'w') as fp:
        fp.write(msg)

testmod()
