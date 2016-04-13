"""
Tests for enum support.
"""

from __future__ import print_function

import enum

import numba.unittest_support as unittest
from numba import jit

from .support import TestCase, tag
from .enum_usecases import *


def compare_usecase(a, b):
    return a == b, a != b, a is b

def constant_usecase(a):
    return a is Color.red

def return_usecase(a, b, pred):
    return a if pred else b


class TestEnum(TestCase):
    pairs = [
        (Color.red, Color.red),
        (Color.red, Color.green),
        (Shake.mint, Shake.vanilla),
        (Planet.VENUS, Planet.MARS),
        (Planet.EARTH, Planet.EARTH),
        ]

    def test_compare(self):
        pyfunc = compare_usecase
        cfunc = jit(nopython=True)(pyfunc)

        for args in self.pairs:
            self.assertPreciseEqual(pyfunc(*args), cfunc(*args))

    def test_return(self):
        """
        Passing and returning enum members.
        """
        pyfunc = return_usecase
        cfunc = jit(nopython=True)(pyfunc)

        for pair in self.pairs:
            for pred in (True, False):
                args = pair + (pred,)
                self.assertIs(pyfunc(*args), cfunc(*args))

    def test_constant(self):
        pyfunc = constant_usecase
        cfunc = jit(nopython=True)(pyfunc)

        for arg in [Color.red, Color.green]:
            self.assertPreciseEqual(pyfunc(arg), cfunc(arg))


if __name__ == '__main__':
    unittest.main()
