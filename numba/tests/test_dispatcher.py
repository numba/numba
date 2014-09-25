from __future__ import print_function, division, absolute_import

import numpy

from numba import unittest_support as unittest
from numba.special import typeof
from numba import utils, vectorize, types, jit
from .support import TestCase


def dummy(x):
    return x


def add(x, y):
    return x + y


class TestDispatcher(TestCase):
    def test_numba_interface(self):
        """
        Check that vectorize can accept a decorated object.
        """
        vectorize('f8(f8)')(jit(dummy))

    def test_no_argument(self):
        @jit
        def foo():
            return 1

        # Just make sure this doesn't crash
        foo()


    def test_inspect_types(self):
        @jit
        def foo(a, b):
            return a + b

        foo(1, 2)
        # Exercise the method
        foo.inspect_types(utils.StringIO())

    def test_coerce_input_types(self):
        # Issue #486: do not allow unsafe conversions if we can still
        # compile other specializations.
        c_add = jit(nopython=True)(add)
        self.assertPreciseEqual(c_add(123, 456), add(123, 456))
        self.assertPreciseEqual(c_add(12.3, 45.6), add(12.3, 45.6))
        self.assertPreciseEqual(c_add(12.3, 45.6j), add(12.3, 45.6j))
        self.assertPreciseEqual(c_add(12300000000, 456), add(12300000000, 456))

        # Now force compilation of only a single specialization
        c_add = jit('(i4, i4)', nopython=True)(add)
        self.assertPreciseEqual(c_add(123, 456), add(123, 456))
        # Implicit (unsafe) conversion of float to int
        self.assertPreciseEqual(c_add(12.3, 45.6), add(12, 45))
        with self.assertRaises(TypeError):
            # Implicit conversion of complex to int disallowed
            c_add(12.3, 45.6j)

    def test_ambiguous_new_version(self):
        """Test compiling new version in an ambiguous case
        """

        @jit
        def foo(a, b):
            return a + b

        INT = 1
        FLT = 1.5
        self.assertAlmostEqual(foo(INT, FLT), INT + FLT)
        self.assertEqual(len(foo.overloads), 1)
        self.assertAlmostEqual(foo(FLT, INT), FLT + INT)
        self.assertEqual(len(foo.overloads), 2)
        self.assertAlmostEqual(foo(FLT, FLT), FLT + FLT)
        self.assertEqual(len(foo.overloads), 3)
        # The following call is ambiguous because (int, int) can resolve
        # to (float, int) or (int, float) with equal weight.
        self.assertAlmostEqual(foo(1, 1), INT + INT)
        self.assertEqual(len(foo.overloads), 4, "didn't compile a new "
                                                "version")


if __name__ == '__main__':
    unittest.main()
