from __future__ import print_function, absolute_import, division
from ctypes import *
from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types


proc = CDLL(None)

c_sin = proc.sin
c_sin.argtypes = [c_double]
c_sin.restype = c_double


def use_c_sin(x):
    return c_sin(x)


class TestCTypes(unittest.TestCase):
    def test_c_sin(self):
        pyfunc = use_c_sin
        cres = compile_isolated(use_c_sin, [types.double])
        cfunc = cres.entry_point
        x = 3.14
        self.assertEqual(pyfunc(x), cfunc(x))

if __name__ == '__main__':
    unittest.main()

