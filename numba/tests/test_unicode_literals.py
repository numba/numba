from __future__ import print_function, unicode_literals

import sys

import numba.unittest_support as unittest
from numba import jit
from .support import TestCase


def docstring_usecase():
    """\u00e9"""
    return 1


@unittest.skipIf(sys.version_info >= (3,), "Python 2-specific test")
class TestFutureUnicodeLiterals(TestCase):
    """
    Test issues with unicode_literals on Python 2.
    """

    def test_docstring(self):
        """
        Test non-ASCII docstring (issue #1908).
        """
        cfunc = jit(nopython=True)(docstring_usecase)
        self.assertPreciseEqual(cfunc(), 1)


if __name__ == '__main__':
    unittest.main()
