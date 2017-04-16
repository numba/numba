"""
Test functions that are almost empty

Related issues:
- https://github.com/numba/numba/issues/614

"""
from __future__ import print_function, absolute_import
from numba import jit
from numba import unittest_support as unittest


def return_one():
    return 1


class TestEmptyFunc(unittest.TestCase):
    def test_issue_614(self):
        self.assertEqual(jit(return_one)(), 1)


if __name__ == '__main__':
    unittest.main()

