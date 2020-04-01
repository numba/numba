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
    def test_add(self):
        @njit(target='dppy')
        def add(a, b):
            c = np.add(a, b)
            return c

        N = 10
        a = np.array(np.random.random(N), dtype=np.float32)
        b = np.array(np.random.random(N), dtype=np.float32)

        c = add(a, b)
        d = a + b

        self.assertTrue(np.all(c == d))


if __name__ == '__main__':
    unittest.main()
