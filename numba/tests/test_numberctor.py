from __future__ import print_function, absolute_import, division

from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types
from .support import TestCase


def doint(a):
    return int(a)


def dofloat(a):
    return float(a)


def docomplex(a):
    return complex(a)


def docomplex2(a, b):
    return complex(a, b)


class TestNumberCtor(TestCase):
    def test_int(self):
        pyfunc = doint

        x_types = [
            types.int32, types.int64, types.float32, types.float64
        ]
        x_values = [1, 1000, 12.2, 23.4]

        for ty, x in zip(x_types, x_values):
            cres = compile_isolated(pyfunc, [ty])
            cfunc = cres.entry_point
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

    def test_float(self):
        pyfunc = dofloat

        x_types = [
            types.int32, types.int64, types.float32, types.float64
        ]
        x_values = [1, 1000, 12.2, 23.4]

        for ty, x in zip(x_types, x_values):
            cres = compile_isolated(pyfunc, [ty])
            cfunc = cres.entry_point
            self.assertPreciseEqual(pyfunc(x), cfunc(x),
                prec='single' if ty is types.float32 else 'exact')

    def test_complex(self):
        pyfunc = docomplex

        x_types = [
            types.int32, types.int64, types.float32, types.float64,
            types.complex64, types.complex128,
        ]
        x_values = [1, 1000, 12.2, 23.4, 1.5-5j, 1-4.75j]

        for ty, x in zip(x_types, x_values):
            cres = compile_isolated(pyfunc, [ty])
            cfunc = cres.entry_point
            got = cfunc(x)
            expected = pyfunc(x)
            self.assertPreciseEqual(pyfunc(x), cfunc(x),
                prec='single' if ty is types.float32 else 'exact')

    def test_complex2(self):
        pyfunc = docomplex2

        x_types = [
            types.int32, types.int64, types.float32, types.float64
        ]
        x_values = [1, 1000, 12.2, 23.4]
        y_values = [x - 3 for x in x_values]

        for ty, x, y in zip(x_types, x_values, y_values):
            cres = compile_isolated(pyfunc, [ty, ty])
            cfunc = cres.entry_point
            self.assertPreciseEqual(pyfunc(x, y), cfunc(x, y),
                prec='single' if ty is types.float32 else 'exact')


if __name__ == '__main__':
    unittest.main()
