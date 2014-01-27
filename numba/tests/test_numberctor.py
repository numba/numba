from __future__ import print_function, absolute_import, division
from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types


def doint(a):
    return int(a)


def dofloat(a):
    return float(a)


def docomplex(a):
    return complex(a)


def docomplex2(a, b):
    return complex(a, b)


class TestNumberCtor(unittest.TestCase):
    def test_int(self):
        pyfunc = doint

        x_types = [
            types.int32, types.int64, types.float32, types.float64
        ]
        x_values = [1, 1000, 12.2, 23.4]

        for ty, x in zip(x_types, x_values):
            cres = compile_isolated(pyfunc, [ty])
            cfunc = cres.entry_point
            self.assertEqual(pyfunc(x), cfunc(x))

    def test_float(self):
        pyfunc = dofloat

        x_types = [
            types.int32, types.int64, types.float32, types.float64
        ]
        x_values = [1, 1000, 12.2, 23.4]

        for ty, x in zip(x_types, x_values):
            cres = compile_isolated(pyfunc, [ty])
            cfunc = cres.entry_point
            self.assertAlmostEqual(pyfunc(x), cfunc(x), places=6)

    def test_complex(self):
        pyfunc = docomplex

        x_types = [
            types.int32, types.int64, types.float32, types.float64
        ]
        x_values = [1, 1000, 12.2, 23.4]

        for ty, x in zip(x_types, x_values):
            cres = compile_isolated(pyfunc, [ty])
            cfunc = cres.entry_point
            self.assertAlmostEqual(pyfunc(x), cfunc(x), places=6)

    def test_complex2(self):
        pyfunc = docomplex2

        x_types = [
            types.int32, types.int64, types.float32, types.float64
        ]
        x_values = [1, 1000, 12.2, 23.4]

        for ty, x in zip(x_types, x_values):
            cres = compile_isolated(pyfunc, [ty, ty])
            cfunc = cres.entry_point
            self.assertAlmostEqual(pyfunc(x, x), cfunc(x, x), places=6)


if __name__ == '__main__':
    unittest.main()
