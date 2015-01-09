"""
Test problems in nested calls.
Usually due to invalid type conversion between function boundaries.

"""
from __future__ import print_function, division, absolute_import
from numba import njit
from numba import unittest_support as unittest


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


if __name__ == '__main__':
    unittest.main()
