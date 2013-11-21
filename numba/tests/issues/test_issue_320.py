# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from math import log, sin, sqrt
import unittest

import numpy as np

from numba import jit, int_

values = np.arange(20, dtype='i')

class TestLoopCarriedDep(unittest.TestCase):

    def test_simple_ish(self):
        @jit(argtypes=(int_[:],))
        def f(vals):
            x = 1.
            prev = 1.
            s = 0.

            for v in vals:
                if v:
                    s += log(x / prev)
                    prev = x

            for v in vals:
                if v:
                    prev = x
                    s += log(x / prev)

            res = sqrt(s / 2)
            return res

        self.assertAlmostEqual(f(values), f.py_func(values))

    def test_less_simple(self):
        @jit(argtypes=(int_[:],))
        def f(vals):
            x = 1.
            prev = 1.
            s = 0.

            for v in vals:
                if v:
                    s += log(x / prev)
                    prev = x

                if v < 4:
                    prev -= 0.1
                if x < 5:
                    s += sin(s)

            for v in vals:
                if v:
                    prev = x
                    s += log(x / prev)

                for i in range(10):
                    prev = sin(prev)

            res = sqrt(s / 2)
            return res

        self.assertAlmostEqual(f(values), f.py_func(values))


if __name__ == '__main__':
    unittest.main()