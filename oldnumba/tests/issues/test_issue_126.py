# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import


from numba import autojit
import math
import unittest

@autojit(nopython=True)
def nopython_abs(x):
    return abs(x)

@autojit(nopython=True)
def nopython_sin(x):
    return math.sin(x)

@autojit(nopython=True)
def nopython_cos(x):
    return math.cos(x)

@autojit(nopython=True)
def nopython_log(x):
    return math.log(x)

@autojit(nopython=True)
def nopython_exp(x):
    return math.exp(x)

class Test(unittest.TestCase):
    def test_nopython_abs(self):
        x = -1
        y = nopython_abs(x)
        self.assertAlmostEqual(y,  abs(x))

    def test_nopython_sin(self):
        x = -1
        y = nopython_sin(x)
        self.assertAlmostEqual(y, math.sin(x))

    def test_nopython_cos(self):
        x = -1
        y = nopython_cos(x)
        self.assertAlmostEqual(y, math.cos(x))

    def test_nopython_log(self):
        x = 10
        y = nopython_log(x)
        self.assertAlmostEqual(y, math.log(x))

    def tet_nopython_exp(self):
        x = 10
        y = nopython_exp(x)
        self.assertAlmostEqual(y, math.exp(x))

if __name__ == '__main__':
    unittest.main()

