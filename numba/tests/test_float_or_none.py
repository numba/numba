# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest
from numba import jit, float_, double

def f(x):
    return x

def g(x):
    if x > 2.0:
        x = None
    return x

class TestFloatOrNone(unittest.TestCase):

    def test_float(self):
        f_float = jit(float_(float_))(f)
        assert f_float(None) == None
        assert f_float(2.0) == 2.0

    def test_double(self):
        f_double = jit(double(double))(f)
        assert f_double(None) == None
        assert f_double(2.0) == 2.0

    def test_float_assign_none(self):
        assign = jit(double(double))(g)
        assert assign(3.0) == None
        assert assign(1.0) == 1.0

if __name__ == '__main__':
    unittest.main()