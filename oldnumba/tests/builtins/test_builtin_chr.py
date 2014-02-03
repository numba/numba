# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import autojit

@autojit(nopython=True)
def test_chr(x):
    return chr(x)

assert test_chr(97) == b'a'