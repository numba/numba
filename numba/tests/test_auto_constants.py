from __future__ import print_function, division, absolute_import

import math
import sys

import numpy as np

from numba import unittest_support as unittest
from numba.compiler import compile_isolated


class TestAutoConstants(unittest.TestCase):
    def test_numpy_nan(self):
        def pyfunc():
            return np.nan

        cres = compile_isolated(pyfunc, ())
        cfunc = cres.entry_point

        self.assertTrue(math.isnan(pyfunc()))
        self.assertTrue(math.isnan(cfunc()))

    def test_sys_constant(self):
        def pyfunc():
            return sys.hexversion

        cres = compile_isolated(pyfunc, ())
        cfunc = cres.entry_point

        self.assertEqual(pyfunc(), cfunc())


if __name__ == '__main__':
    unittest.main()

