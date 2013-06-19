# -*- coding: utf-8 -*-

"""
>>> test_ord('hello')
101L
"""

from __future__ import print_function, division, absolute_import

from numba import *

@autojit(nopython=True)
def test_ord(s):
    return ord(s[1])

if __name__ == '__main__':
    import numba
    numba.testing.testmod()