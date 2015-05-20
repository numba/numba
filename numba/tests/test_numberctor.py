from __future__ import print_function, absolute_import, division

import numpy as np

from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba import jit, types

from .support import TestCase


def doint(a):
    return int(a)


def dofloat(a):
    return float(a)


def docomplex(a):
    return complex(a)


def docomplex2(a, b):
    return complex(a, b)


def complex_calc(a):
    z = complex(a)
    return z.real ** 2 + z.imag ** 2


def complex_calc2(a, b):
    z = complex(a, b)
    return z.real ** 2 + z.imag ** 2


def converter(tp):
    def f(a):
        return tp(a)
    return f


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

        # Check that complex(float32) really creates a complex64,
        # by checking the accuracy of computations.
        pyfunc = complex_calc
        x = 1.0 + 2**-50
        cres = compile_isolated(pyfunc, [types.float32])
        cfunc = cres.entry_point
        self.assertPreciseEqual(cfunc(x), 1.0)
        # Control (complex128)
        cres = compile_isolated(pyfunc, [types.float64])
        cfunc = cres.entry_point
        self.assertGreater(cfunc(x), 1.0)

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

        # Check that complex(float32, float32) really creates a complex64,
        # by checking the accuracy of computations.
        pyfunc = complex_calc2
        x = 1.0 + 2**-50
        cres = compile_isolated(pyfunc, [types.float32, types.float32])
        cfunc = cres.entry_point
        self.assertPreciseEqual(cfunc(x, x), 2.0)
        # Control (complex128)
        cres = compile_isolated(pyfunc, [types.float64, types.float32])
        cfunc = cres.entry_point
        self.assertGreater(cfunc(x, x), 2.0)

    def check_type_converter(self, tp, np_type, values):
        pyfunc = converter(tp)
        cfunc = jit(nopython=True)(pyfunc)
        if issubclass(np_type, np.integer):
            # Converting from a Python int to a small Numpy int on 32-bit
            # builds can raise "OverflowError: Python int too large to
            # convert to C long".  Work around by going through a large
            # Numpy int first.
            np_converter = lambda x: np_type(np.int64(x))
        else:
            np_converter = np_type
        for val in values:
            expected = np_converter(val)
            got = cfunc(val)
            self.assertPreciseEqual(got, expected)

    def check_number_types(self, tp_factory):
        values = [0, 1, -1, 100003, 10000000000007, -100003, -10000000000007,
                  1.5, -3.5]
        for tp_name in ('int8', 'int16', 'int32', 'int64',
                        'uint8', 'uint16', 'uint32', 'uint64',
                        'intc', 'uintc', 'intp', 'uintp',
                        'float32', 'float64', 'bool_'):
            np_type = getattr(np, tp_name)
            tp = tp_factory(tp_name)
            self.check_type_converter(tp, np_type, values)
        values.append(1.5+3j)
        for tp_name in ('complex64', 'complex128'):
            np_type = getattr(np, tp_name)
            tp = tp_factory(tp_name)
            self.check_type_converter(tp, np_type, values)

    def test_numba_types(self):
        """
        Test explicit casting to Numba number types.
        """
        def tp_factory(tp_name):
            return getattr(types, tp_name)
        self.check_number_types(tp_factory)

    def test_numpy_types(self):
        """
        Test explicit casting to Numpy number types.
        """
        def tp_factory(tp_name):
            return getattr(np, tp_name)
        self.check_number_types(tp_factory)


if __name__ == '__main__':
    unittest.main()
