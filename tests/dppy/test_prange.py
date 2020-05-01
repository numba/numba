#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy, njit, prange
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase

import dppy.core as ocldrv


class TestPrange(DPPYTestCase):
    def test_one_prange(self):
        @njit(parallel={'spirv':True})
        def f(a, b):
            for i in prange(4):
                b[i, 0] = a[i, 0] * 10

        m = 8
        n = 8
        a = np.ones((m, n))
        b = np.ones((m, n))

        f(a, b)

        for i in range(4):
            self.assertTrue(b[i, 0] == a[i, 0] * 10)


    def test_nested_prange(self):
        @njit(parallel={'spirv':True})
        def f(a, b):
            # dimensions must be provided as scalar
            m, n = a.shape
            for i in prange(m):
                for j in prange(n):
                    b[i, j] = a[i, j] * 10

        m = 8
        n = 8
        a = np.ones((m, n))
        b = np.ones((m, n))

        f(a, b)
        self.assertTrue(np.all(b == 10))


    def test_multiple_prange(self):
        @njit(parallel={'spirv':True})
        def f(a, b):
            # dimensions must be provided as scalar
            m, n = a.shape
            for i in prange(m):
                val = 10
                for j in prange(n):
                    b[i, j] = a[i, j] * val


            for i in prange(m):
                for j in prange(n):
                    a[i, j] = a[i, j] * 10

        m = 8
        n = 8
        a = np.ones((m, n))
        b = np.ones((m, n))

        f(a, b)
        self.assertTrue(np.all(b == 10))
        self.assertTrue(np.all(a == 10))


    def test_three_prange(self):
        @njit(parallel={'spirv':True})
        def f(a, b):
            # dimensions must be provided as scalar
            m, n, o = a.shape
            for i in prange(m):
                val = 10
                for j in prange(n):
                    constant = 2
                    for k in prange(o):
                        b[i, j, k] = a[i, j, k] * (val + constant)

        m = 8
        n = 8
        o = 8
        a = np.ones((m, n, o))
        b = np.ones((m, n, o))

        f(a, b)
        self.assertTrue(np.all(b == 12))



if __name__ == '__main__':
    unittest.main()
