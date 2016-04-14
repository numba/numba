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
    return a == b, a != b, a is b, a is not b

def global_usecase(a):
    # Lookup of a enum member on its class
    return a is Color.red

def make_constant_usecase(const):
    def constant_usecase(a):
        return a is const
    return constant_usecase

def return_usecase(a, b, pred):
    return a if pred else b

def int_coerce_usecase(x):
    # Implicit coercion of intenums to ints
    if x > RequestError.internal_error:
        return x - RequestError.not_found
    else:
        return x + Shape.circle


class TestEnum(TestCase):
    values = [Color.red, Color.green]

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

    def check_constant_usecase(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)

        for arg in self.values:
            self.assertPreciseEqual(pyfunc(arg), cfunc(arg))

    def test_constant(self):
        self.check_constant_usecase(global_usecase)
        self.check_constant_usecase(make_constant_usecase(self.values[0]))


class TestIntEnum(TestEnum):
    values = [Shape.circle, Shape.square]

    pairs = [
        (Shape.circle, Shape.circle),
        (Shape.circle, Shape.square),
        (RequestError.not_found, RequestError.not_found),
        (RequestError.internal_error, RequestError.not_found),
        ]

    def test_int_coerce(self):
        pyfunc = int_coerce_usecase
        cfunc = jit(nopython=True)(pyfunc)

        for arg in [300, 450, 550]:
            self.assertPreciseEqual(pyfunc(arg), cfunc(arg))


if __name__ == '__main__':
    unittest.main()
