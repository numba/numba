from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppl, njit
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase


class TestNumpy_floating_functions(DPPLTestCase):
    def test_isfinite(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.isfinite(a)
            return c

        test_arr = [np.log(-1.),1.,np.log(0)]
        input_arr = np.asarray(test_arr, dtype=np.float32)

        c = f(input_arr)
        d = np.isfinite(input_arr)
        self.assertTrue(np.all(c == d))


    def test_isinf(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.isinf(a)
            return c

        test_arr = [np.log(-1.),1.,np.log(0)]
        input_arr = np.asarray(test_arr, dtype=np.float32)

        c = f(input_arr)
        d = np.isinf(input_arr)
        self.assertTrue(np.all(c == d))

    def test_isnan(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.isnan(a)
            return c

        test_arr = [np.log(-1.),1.,np.log(0)]
        input_arr = np.asarray(test_arr, dtype=np.float32)

        c = f(input_arr)
        d = np.isnan(input_arr)
        self.assertTrue(np.all(c == d))


    def test_floor(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.floor(a)
            return c

        input_arr = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])

        c = f(input_arr)
        d = np.floor(input_arr)
        self.assertTrue(np.all(c == d))


    def test_ceil(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.ceil(a)
            return c

        input_arr = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])

        c = f(input_arr)
        d = np.ceil(input_arr)
        self.assertTrue(np.all(c == d))


    def test_trunc(self):
        @njit(parallel={'offload':True})
        def f(a):
            c = np.trunc(a)
            return c

        input_arr = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])

        c = f(input_arr)
        d = np.trunc(input_arr)
        self.assertTrue(np.all(c == d))



if __name__ == '__main__':
    unittest.main()
