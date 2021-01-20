import math
import sys

import numpy as np

from numba import njit
from numba.core.compiler import compile_isolated
import numba.tests.usecases as uc
import unittest


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

    def test_module_string_constant(self):
        @njit
        def f():
            return uc._GLOBAL_STR
        self.assertEqual(f(), f.py_func())


if __name__ == '__main__':
    unittest.main()
