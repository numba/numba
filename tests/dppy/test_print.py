#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy, njit, prange
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase

import dppy as ocldrv


@unittest.skipUnless(ocldrv.has_gpu_device, 'test only on GPU system')
class TestPrint(DPPYTestCase):
    def test_print_prange(self):
        @njit(parallel={'spirv':True})
        def f(a, b):
            for i in prange(4):
                print("before at:", i, b[i, 0])
                b[i, 0] = a[i, 0] * 10
                print("after at:", i, b[i, 0])

        m = 8
        n = 8
        a = np.ones((m, n))
        b = np.ones((m, n))

        f(a, b)

        for i in range(4):
            self.assertTrue(b[i, 0] == a[i, 0] * 10)

    def test_print_dppy_kernel(self):
        @dppy.func
        def g(a):
            print("value of a:", a)
            return a + 1

        @dppy.kernel
        def f(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i])
            print("value of b at:", i, "is", b[i])

        N = 10

        a = np.ones(N)
        b = np.ones(N)

        with ocldrv.igpu_context(0) as device_env:
            f[N, dppy.DEFAULT_LOCAL_SIZE](a, b)


if __name__ == '__main__':
    unittest.main()
