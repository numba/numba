#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy, njit, prange
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase

import dppy.core as ocldrv


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
        device_env = None

        try:
            device_env = ocldrv.runtime.get_gpu_device()
            print("Selected GPU device")
        except:
            print("GPU device not found")
            exit()

        a = np.ones(N)
        b = np.ones(N)

        f[device_env, N](a, b)


if __name__ == '__main__':
    unittest.main()
