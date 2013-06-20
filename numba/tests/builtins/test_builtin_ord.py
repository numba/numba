# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import autojit

@autojit(nopython=True)
def test_ord(s):
    return ord(s[1])

assert test_ord('hello') == 101