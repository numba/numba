from __future__ import print_function

from numba import njit
from .support import TestCase, unittest


class TestUsecases(TestCase):
    def test_try_inner_raise(self):
        @njit
        def inner(x):
            if x:
                raise ZeroDivisionError

        @njit
        def udt(x):
            try:
                inner(x)
                return "not raised"
            except:             # noqa: E722
                return "caught"

        self.assertEqual(udt(False), "not raised")
        self.assertEqual(udt(True), "caught")


if __name__ == '__main__':
    unittest.main()
