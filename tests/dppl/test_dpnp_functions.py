#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppl, njit
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase


def test_for_different_datatypes(fn, test_fn, size, arg_count):
    tys = [np.int32, np.uint32, np.int64, np.uint64, np.float, np.double]
    if arg_count == 1:
        for ty in tys:
            a = np.array(np.random.random(size), dtype=ty)
            c = fn(a)
            d = test_fn(a)
            max_abs_err = c - d
            if not (max_abs_err < 1e-4):
                return False

    return True

class Testdpnp_functions(DPPLTestCase):
    N = 10

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)

    def test_sum(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.sum(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.sum, 10, 1))

    def test_prod(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.prod(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.prod, 10, 1))

    def test_argmax(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.argmax(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.argmax, 10, 1))

    def test_argmin(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.argmin(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.argmin, 10, 1))




if __name__ == '__main__':
    unittest.main()
