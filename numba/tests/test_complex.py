from __future__ import print_function, absolute_import, division

import cmath
import itertools
import math

from numba import unittest_support as unittest
from numba.compiler import compile_isolated, Flags, utils
from numba import types
from .support import TestCase

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()


def real_usecase(x):
    return x.real

def imag_usecase(x):
    return x.imag

def conjugate_usecase(x):
    return x.conjugate()


def isfinite_usecase(x):
    return cmath.isfinite(x)

def isinf_usecase(x):
    return cmath.isinf(x)

def isnan_usecase(x):
    return cmath.isnan(x)

def rect_usecase(r, phi):
    return cmath.rect(r, phi)


class BaseComplexTest(object):

    def basic_values(self):
        reals = [-0.0, +0.0, 1, -1,
                 float('-inf'), float('+inf'), float('nan')]
        return [complex(x, y) for x, y in itertools.product(reals, reals)]

    def more_values(self):
        reals = [-0.0, +0.0, 1, -1, +1.5, -3.5, -math.pi, +math.pi,
                 float('-inf'), float('+inf'), float('nan')]
        return [complex(x, y) for x, y in itertools.product(reals, reals)]

    def run_unary(self, pyfunc, x_types, x_values, flags=enable_pyobj_flags,
                  prec='exact'):
        for tx in x_types:
            cr = compile_isolated(pyfunc, [tx], flags=flags)
            cfunc = cr.entry_point
            actual_prec = 'single' if tx == types.float32 else prec
            for vx in x_values:
                got = cfunc(vx)
                expected = pyfunc(vx)
                msg = 'for input %r' % (vx,)
                self.assertPreciseEqual(got, expected, prec=prec, msg=msg)

    def run_binary(self, pyfunc, value_types, values,
                   flags=enable_pyobj_flags, prec='exact'):
        for tx, ty in value_types:
            cr = compile_isolated(pyfunc, [tx, ty], flags=flags)
            cfunc = cr.entry_point
            actual_prec = 'single' if types.float32 in (tx, ty) else prec
            for vx, vy in values:
                got = cfunc(vx, vy)
                expected = pyfunc(vx, vy)
                msg = 'for input %r' % ((vx, vy),)
                self.assertPreciseEqual(got, expected, prec=prec, msg=msg)


class TestComplex(BaseComplexTest, TestCase):

    def test_real(self, flags=enable_pyobj_flags):
        self.run_unary(real_usecase, [types.complex64, types.complex128],
                       self.basic_values(), flags=flags)
        self.run_unary(real_usecase, [types.int8, types.int64],
                       [1, 0, -3], flags=flags)
        self.run_unary(real_usecase, [types.float32, types.float64],
                       [1.5, -0.5], flags=flags)

    def test_real_npm(self):
        self.test_real(flags=no_pyobj_flags)

    def test_imag(self, flags=enable_pyobj_flags):
        self.run_unary(imag_usecase, [types.complex64, types.complex128],
                       self.basic_values(), flags=flags)
        self.run_unary(imag_usecase, [types.int8, types.int64],
                       [1, 0, -3], flags=flags)
        self.run_unary(imag_usecase, [types.float32, types.float64],
                       [1.5, -0.5], flags=flags)

    def test_imag_npm(self):
        self.test_imag(flags=no_pyobj_flags)

    def test_conjugate(self, flags=enable_pyobj_flags):
        self.run_unary(conjugate_usecase, [types.complex64, types.complex128],
                       self.basic_values(), flags=flags)
        self.run_unary(conjugate_usecase, [types.int8, types.int64],
                       [1, 0, -3], flags=flags)
        self.run_unary(conjugate_usecase, [types.float32, types.float64],
                       [1.5, -0.5], flags=flags)

    def test_conjugate_npm(self):
        self.test_conjugate(flags=no_pyobj_flags)


class TestCMath(BaseComplexTest, TestCase):
    """
    Tests for cmath module support.
    """

    def check_predicate_func(self, pyfunc, flags):
        self.run_unary(pyfunc, [types.complex64, types.complex128],
                       self.basic_values(), flags=flags)

    def test_isnan(self, flags=enable_pyobj_flags):
        self.check_predicate_func(isnan_usecase, enable_pyobj_flags)

    def test_isnan_npm(self):
        self.check_predicate_func(isnan_usecase, no_pyobj_flags)

    def test_isinf(self, flags=enable_pyobj_flags):
        self.check_predicate_func(isinf_usecase, enable_pyobj_flags)

    def test_isinf_npm(self):
        self.check_predicate_func(isinf_usecase, no_pyobj_flags)

    @unittest.skipIf(utils.PYVERSION < (3, 2), "needs Python 3.2+")
    def test_isfinite(self, flags=enable_pyobj_flags):
        self.check_predicate_func(isfinite_usecase, enable_pyobj_flags)

    @unittest.skipIf(utils.PYVERSION < (3, 2), "needs Python 3.2+")
    def test_isfinite_npm(self):
        self.check_predicate_func(isfinite_usecase, no_pyobj_flags)

    def test_rect(self, flags=enable_pyobj_flags):
        values = [(z.real, z.imag) for z in self.more_values()
                  if not math.isinf(z.imag) or z.real == 0]
        value_types = [(types.float64, types.float64)]
        self.run_binary(rect_usecase, value_types, values, flags=flags,
                        prec='single')

    def test_rect_npm(self):
        self.test_rect(flags=no_pyobj_flags)


if __name__ == '__main__':
    unittest.main()
