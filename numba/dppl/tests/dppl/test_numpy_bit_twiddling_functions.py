#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppl, njit
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase


class TestNumpy_bit_twiddling_functions(DPPLTestCase):
    def test_bitwise_and(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.bitwise_and(a, b)
            return c

        a = np.array([2,5,255])
        b = np.array([3,14,16])

        c = f(a, b)
        d = np.bitwise_and(a, b)
        self.assertTrue(np.all(c == d))


    def test_bitwise_or(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.bitwise_or(a, b)
            return c

        a = np.array([2,5,255])
        b = np.array([4,4,4])

        c = f(a, b)
        d = np.bitwise_or(a, b)
        self.assertTrue(np.all(c == d))


    def test_bitwise_xor(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.bitwise_xor(a, b)
            return c

        a = np.array([2,5,255])
        b = np.array([4,4,4])

        c = f(a, b)
        d = np.bitwise_xor(a, b)
        self.assertTrue(np.all(c == d))


    def test_bitwise_not(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.bitwise_not(a)
            return c

        a = np.array([2,5,255])

        c = f(a)
        d = np.bitwise_not(a)
        self.assertTrue(np.all(c == d))


    def test_invert(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.invert(a)
            return c

        a = np.array([2,5,255])

        c = f(a)
        d = np.invert(a)
        self.assertTrue(np.all(c == d))


    def test_left_shift(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.left_shift(a, b)
            return c

        a = np.array([2,3,4])
        b = np.array([1,2,3])

        c = f(a, b)
        d = np.left_shift(a, b)
        self.assertTrue(np.all(c == d))


    def test_right_shift(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.right_shift(a, b)
            return c

        a = np.array([2,3,4])
        b = np.array([1,2,3])

        c = f(a, b)
        d = np.right_shift(a, b)
        self.assertTrue(np.all(c == d))


if __name__ == '__main__':
    unittest.main()
