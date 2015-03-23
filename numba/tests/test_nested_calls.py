"""
Test problems in nested calls.
Usually due to invalid type conversion between function boundaries.
"""

from __future__ import print_function, division, absolute_import

from numba import njit
from numba import unittest_support as unittest


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



class TestNestedCall(unittest.TestCase):

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
        cfunc = njit(f)
        self.assertEqual(cfunc(1, 2, 3), f(1, 2, 3))
        self.assertEqual(cfunc(1, y=2, z=3), f(1, 2, 3))

    def test_default_args(self):
        """
        Test a nested function call using default argument values.
        """
        cfunc = njit(g)
        self.assertEqual(cfunc(1, 2, 3), g(1, 2, 3))
        self.assertEqual(cfunc(1, y=2, z=3), g(1, 2, 3))


if __name__ == '__main__':
    unittest.main()
