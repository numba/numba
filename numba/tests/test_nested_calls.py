"""
Test problems in nested calls.
Usually due to invalid type conversion between function boundaries.
"""

from __future__ import print_function, division, absolute_import

from numba import njit
from numba import unittest_support as unittest
from .support import TestCase


@njit
def f_inner(a, b, c):
    return a, b, c

def f(x, y, z):
    return f_inner(x, c=y, b=z)

@njit
def g_inner(a, b=2, c=3):
    return a, b, c

def g(x, y, z):
    return g_inner(x, b=y), g_inner(a=z, c=x)

@njit
def star_inner(a=5, *b):
    return a, b

def star(x, y, z):
    return star_inner(a=x), star_inner(x, y, z)

def star_call(x, y, z):
    return star_inner(x, *y), star_inner(*z)


class TestNestedCall(TestCase):

    def compile_func(self, pyfunc):
        def check(*args, **kwargs):
            expected = pyfunc(*args, **kwargs)
            result = f(*args, **kwargs)
            self.assertPreciseEqual(result, expected)
        f = njit(pyfunc)
        return f, check

    def test_boolean_return(self):
        @njit
        def inner(x):
            return not x

        @njit
        def outer(x):
            if inner(x):
                return True
            else:
                return False

        self.assertFalse(outer(True))
        self.assertTrue(outer(False))

    def test_named_args(self):
        """
        Test a nested function call with named (keyword) arguments.
        """
        cfunc, check = self.compile_func(f)
        check(1, 2, 3)
        check(1, y=2, z=3)

    def test_default_args(self):
        """
        Test a nested function call using default argument values.
        """
        cfunc, check = self.compile_func(g)
        check(1, 2, 3)
        check(1, y=2, z=3)

    def test_star_args(self):
        """
        Test a nested function call to a function with *args in its signature.
        """
        cfunc, check = self.compile_func(star)
        check(1, 2, 3)

    def test_star_call(self):
        """
        Test a function call with a *args.
        """
        cfunc, check = self.compile_func(star_call)
        check(1, (2,), (3,))


if __name__ == '__main__':
    unittest.main()
