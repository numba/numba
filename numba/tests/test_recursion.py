from __future__ import print_function, division, absolute_import

import numpy as np

from numba import unittest_support as unittest
from numba import jit
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
        with self.assertTypingError() as raises:
            self.mod.fib3(10)
        self.assertIn("recursive calls need an explicit signature",
                      str(raises.exception))


class TestMutualRecursion(TestCase):

    def setUp(self):
        from . import recursion_usecases
        self.mod = recursion_usecases

    def test_mutual(self):
        with self.assertTypingError() as raises:
            self.mod.outer_fac(10)
        self.assertIn("mutual recursion not supported",
                      str(raises.exception))


if __name__ == '__main__':
    unittest.main()
