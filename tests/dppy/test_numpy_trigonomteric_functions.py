#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy, njit
from numba.dppy.dppy_driver import driver as ocldrv
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase

class TestNumpy_math_functions(DPPYTestCase):
    N = 10
    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)

    def test_deg2rad(self):
        @njit(target='dppy')
        def f(a):
            c = np.deg2rad(a)
            return c

        c = f(self.a)
        d = np.deg2rad(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_rad2deg(self):
        @njit(target='dppy')
        def f(a):
            c = np.rad2deg(a)
            return c

        c = f(self.a)
        d = np.rad2deg(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_degrees(self):
        @njit(target='dppy')
        def f(a):
            c = np.degrees(a)
            return c

        c = f(self.a)
        d = np.degrees(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_radians(self):
        @njit(target='dppy')
        def f(a):
            c = np.radians(a)
            return c

        c = f(self.a)
        d = np.radians(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


if __name__ == '__main__':
    unittest.main()
