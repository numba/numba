from __future__ import division

import numba

from numba import unittest_support as unittest
from .support import TestCase, tag


class TestNumbaModule(TestCase):
    """
    Test the APIs exposed by the top-level `numba` module.
    """

    def check_member(self, name):
        self.assertTrue(hasattr(numba, name), name)
        self.assertIn(name, numba.__all__)

    @tag('important')
    def test_numba_module(self):
        # jit
        self.check_member("jit")
        self.check_member("vectorize")
        self.check_member("guvectorize")
        self.check_member("njit")
        self.check_member("autojit")
        # errors
        self.check_member("NumbaError")
        self.check_member("TypingError")
        # types
        self.check_member("int32")
        # misc
        numba.__version__  # not in __all__


if __name__ == '__main__':
    unittest.main()
