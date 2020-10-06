#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppl, njit, vectorize
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase


class TestVectorize(DPPLTestCase):
    def test_vectorize(self):

        @vectorize(nopython=True)
        def axy(a, x, y):
            return a * x + y

        @njit(parallel={'offload':True})
        def f(a0, a1):
            return np.cos(axy(a0, np.sin(a1) - 1., 1.) )

        def f_np(a0, a1):
            sin_res = np.sin(a1)
            res = []
            for i in range(len(a0)):
                res.append(axy(a0[i], sin_res[i] - 1., 1.))
            return np.cos(np.array(res))


        A = np.random.random(10)
        B = np.random.random(10)

        expected = f(A, B)
        actual = f_np(A, B)

        max_abs_err = expected.sum() - actual.sum()
        self.assertTrue(max_abs_err < 1e-5)


if __name__ == '__main__':
    unittest.main()
