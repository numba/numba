# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numba import jit
from numpy import zeros

import unittest

@jit()
def test():
    foo = zeros((1,))
    foo[0] = 0

@jit()
def test2():
    foo = [0]
    foo[0] = 0

class TestIssue50(unittest.TestCase):
    def test_1d_arr_setitem(self):
        self.assertEquals(test(), None)

    def test_list_setitem(self):
        self.assertEqual(test2(), None)

if __name__ == "__main__":
    unittest.main()
