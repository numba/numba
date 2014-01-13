# -*- coding: utf-8 -*-

"""
Test binding of autojit methods.
"""

from __future__ import print_function, division, absolute_import

from numba import *

class A(object):
    @autojit
    def a(self, arg):
        return self * arg

    def __mul__(self, other):
        return 10 * other

assert A().a(10) == 100
