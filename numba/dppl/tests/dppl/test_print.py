#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppl, njit, prange
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase

import dpctl


@unittest.skipUnless(dpctl.has_gpu_queues(), 'test only on GPU system')
class TestPrint(DPPLTestCase):
    def test_print_dppl_kernel(self):
        @dppl.func
        def g(a):
            print("value of a:", a)
            return a + 1

        @dppl.kernel
        def f(a, b):
            i = dppl.get_global_id(0)
            b[i] = g(a[i])
            print("value of b at:", i, "is", b[i])

        N = 10

        a = np.ones(N)
        b = np.ones(N)

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            f[N, dppl.DEFAULT_LOCAL_SIZE](a, b)


if __name__ == '__main__':
    unittest.main()
