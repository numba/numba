from __future__ import print_function, division, absolute_import
from numba import unittest_support as unittest
from numba.compiler import compile_isolated
import numpy
import math


class TestNumPyConstants(unittest.TestCase):
    def test_nan(self):
        def pyfunc():
            return numpy.nan

        cres = compile_isolated(pyfunc, ())
        cfunc = cres.entry_point

        self.assertTrue(math.isnan(pyfunc()))
        self.assertTrue(math.isnan(cfunc()))


if __name__ == '__main__':
    unittest.main()
