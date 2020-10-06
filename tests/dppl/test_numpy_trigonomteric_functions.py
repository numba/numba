#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppl, njit
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase


class TestNumpy_math_functions(DPPLTestCase):
    N = 10
    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)

    def test_sin(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.sin(a)
            return c

        c = f(self.a)
        d = np.sin(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_cos(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.cos(a)
            return c

        c = f(self.a)
        d = np.cos(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_tan(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.tan(a)
            return c

        c = f(self.a)
        d = np.tan(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_arcsin(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.arcsin(a)
            return c

        c = f(self.a)
        d = np.arcsin(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_arccos(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.arccos(a)
            return c

        c = f(self.a)
        d = np.arccos(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_arctan(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.arctan(a)
            return c

        c = f(self.a)
        d = np.arctan(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_arctan2(self):
        @njit(parallel={'offload':True})
        def f(a, b):
            c = np.arctan2(a, b)
            return c

        c = f(self.a, self.b)
        d = np.arctan2(self.a, self.b)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_sinh(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.sinh(a)
            return c

        c = f(self.a)
        d = np.sinh(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_cosh(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.cosh(a)
            return c

        c = f(self.a)
        d = np.cosh(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_tanh(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.tanh(a)
            return c

        c = f(self.a)
        d = np.tanh(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_arcsinh(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.arcsinh(a)
            return c

        c = f(self.a)
        d = np.arcsinh(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_arccosh(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.arccosh(a)
            return c

        input_arr = np.random.randint(1, self.N, size=(self.N))
        c = f(input_arr)
        d = np.arccosh(input_arr)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_arctanh(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.arctanh(a)
            return c

        c = f(self.a)
        d = np.arctanh(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_deg2rad(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.deg2rad(a)
            return c

        c = f(self.a)
        d = np.deg2rad(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


    def test_rad2deg(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.rad2deg(a)
            return c

        c = f(self.a)
        d = np.rad2deg(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-2)

    def test_degrees(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.degrees(a)
            return c

        c = f(self.a)
        d = np.degrees(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-2)

    def test_radians(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.radians(a)
            return c

        c = f(self.a)
        d = np.radians(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


if __name__ == '__main__':
    unittest.main()
