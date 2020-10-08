#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
import numba
from numba import dppl, njit, prange
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase
from numba.tests.support import captured_stdout


class TestPrange(DPPLTestCase):
    def test_one_prange(self):
        @njit(parallel={'offload':True})
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
        @njit(parallel={'offload':True})
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
        @njit(parallel={'offload':True})
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
        @njit(parallel={'offload':True})
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


    def test_two_consequent_prange(self):
        def prange_example():
            n = 10
            a = np.ones((n), dtype=np.float64)
            b = np.ones((n), dtype=np.float64)
            c = np.ones((n), dtype=np.float64)
            for i in prange(n//2):
                a[i] = b[i] + c[i]

            return a

        old_debug = numba.dppl.compiler.DEBUG
        numba.dppl.compiler.DEBUG = 1

        jitted = njit(parallel={'offload':True})(prange_example)
        with captured_stdout() as got:
            jitted_res = jitted()

        res = prange_example()

        numba.dppl.compiler.DEBUG = old_debug

        self.assertEqual(got.getvalue().count('Parfor lowered on DPPL-device'), 2)
        self.assertEqual(got.getvalue().count('Failed to lower parfor on DPPL-device'), 0)
        np.testing.assert_equal(res, jitted_res)


    @unittest.skip('NRT required but not enabled')
    def test_2d_arrays(self):
        def prange_example():
            n = 10
            a = np.ones((n, n), dtype=np.float64)
            b = np.ones((n, n), dtype=np.float64)
            c = np.ones((n, n), dtype=np.float64)
            for i in prange(n//2):
                a[i] = b[i] + c[i]

            return a

        old_debug = numba.dppl.compiler.DEBUG
        numba.dppl.compiler.DEBUG = 1

        jitted = njit(parallel={'offload':True})(prange_example)
        with captured_stdout() as got:
            jitted_res = jitted()

        res = prange_example()

        numba.dppl.compiler.DEBUG = old_debug

        self.assertEqual(got.getvalue().count('Parfor lowered on DPPL-device'), 2)
        self.assertEqual(got.getvalue().count('Failed to lower parfor on DPPL-device'), 0)
        np.testing.assert_equal(res, jitted_res)


if __name__ == '__main__':
    unittest.main()
