from __future__ import print_function, absolute_import, division

import cmath

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


class TestComplex(TestCase):

    def run_unary(self, pyfunc, x_types, x_values, flags=enable_pyobj_flags,
                  prec='exact'):
        for tx in x_types:
            cr = compile_isolated(pyfunc, [tx], flags=flags)
            cfunc = cr.entry_point
            for vx in x_values:
                got = cfunc(vx)
                expected = pyfunc(vx)
                msg = 'for input %r' % (vx,)
                self.assertPreciseEqual(got, expected, prec=prec, msg=msg)

    def test_real(self, flags=enable_pyobj_flags):
        self.run_unary(real_usecase, [types.complex64, types.complex128],
                       [1+1j, -1+1j, float('inf') + 1j, float('nan') + 1j,
                        1 + 1j * float('nan')], flags=flags)

    def test_real_npm(self):
        self.test_real(flags=no_pyobj_flags)

    def test_imag(self, flags=enable_pyobj_flags):
        self.run_unary(imag_usecase, [types.complex64, types.complex128],
                       [1+1j, 1-1j, 1j * float('inf'), 1j * float('nan'),
                        float('nan') + 1j], flags=flags)

    def test_imag_npm(self):
        self.test_imag(flags=no_pyobj_flags)

    def test_conjugate(self, flags=enable_pyobj_flags):
        self.run_unary(conjugate_usecase, [types.complex64, types.complex128],
                       [1+1j, 1-1j, 1j * float('inf'), 1j * float('nan'),
                        float('nan') + 1j], flags=flags)

    def test_conjugate_npm(self):
        self.test_conjugate(flags=no_pyobj_flags)


if __name__ == '__main__':
    unittest.main()
