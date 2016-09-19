from __future__ import print_function, division, absolute_import

import math

from numba import unittest_support as unittest
from .support import TestCase, tag


class TestSelfRecursion(TestCase):

    def setUp(self):
        # Avoid importing this module at toplevel, as it triggers compilation
        # and can therefore fail
        from . import recursion_usecases
        self.mod = recursion_usecases

    def check_fib(self, cfunc):
        self.assertPreciseEqual(cfunc(10), 55)

    @tag('important')
    def test_global_explicit_sig(self):
        self.check_fib(self.mod.fib1)

    def test_inner_explicit_sig(self):
        self.check_fib(self.mod.fib2)

    def test_global_implicit_sig(self):
        self.check_fib(self.mod.fib3)


class TestMutualRecursion(TestCase):

    def setUp(self):
        from . import recursion_usecases
        self.mod = recursion_usecases

    def test_mutual(self):
        expect = math.factorial(10)
        self.assertPreciseEqual(self.mod.outer_fac(10), expect)


if __name__ == '__main__':
    unittest.main()
