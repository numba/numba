#! /usr/bin/env python
# ______________________________________________________________________
'''test_complex

Test Numba's ability to generate code that supports complex numbers.
'''
# ______________________________________________________________________

import unittest

from numba import *
from numba.decorators import jit
from numba.utils import debugout
from numba.llvm_types import _plat_bits
from numba.tests import test_support

import numpy
import itertools

# ______________________________________________________________________

def get_real_fn (in_num):
    return in_num.real

def get_imag_fn (in_num):
    ret_val = in_num.imag
    return ret_val

def get_conj_fn (in_num):
    return in_num.conjugate()

def get_complex_constant_fn ():
    return (3. + 4.j).conjugate()

def prod_sum_fn (coeff, inval, ofs):
    #debugout('prod_sum_fn(): coeff = ', coeff, ', inval = ', inval, ', ofs = ',
    #         ofs)
    ret_val = (coeff * inval) + ofs
    #debugout('prod_sum_fn() ==> ', ret_val)
    return ret_val

# ______________________________________________________________________

class TestComplex (test_support.ByteCodeTestCase):
    def test_get_real_fn (self):
        num0 = 3 + 2j
        num1 = numpy.complex128(num0)
        compiled_get_real_fn = self.jit(argtypes = [complex128])(get_real_fn)
        self.assertEqual(compiled_get_real_fn(num0), 3.)
        self.assertEqual(get_real_fn(num0), compiled_get_real_fn(num0))
        self.assertEqual(compiled_get_real_fn(num1), 3.)
        self.assertEqual(get_real_fn(num1), compiled_get_real_fn(num1))

    def test_get_imag_fn (self):
        num0 = 0 - 2j
        num1 = numpy.complex128(num0)
        compiled_get_imag_fn = self.jit(argtypes = [complex128])(get_imag_fn)
        self.assertEqual(compiled_get_imag_fn(num0), -2.)
        self.assertEqual(get_imag_fn(num0), compiled_get_imag_fn(num0))
        self.assertEqual(compiled_get_imag_fn(num1), -2.)
        self.assertEqual(get_imag_fn(num1), compiled_get_imag_fn(num1))

    @unittest.skipUnless(_plat_bits == 64, 'Complex return values not '
                         'supported on 32-bit systems.')
    def test_get_conj_fn (self):
        num0 = 4 - 1.5j
        num1 = numpy.complex128(num0)
        compiled_get_conj_fn = self.jit(argtypes = [complex128],
                                        restype = complex128)(get_conj_fn)
        self.assertEqual(compiled_get_conj_fn(num0), 4 + 1.5j)
        self.assertEqual(get_conj_fn(num0), compiled_get_conj_fn(num0))
        self.assertEqual(compiled_get_conj_fn(num1), 4 + 1.5j)
        self.assertEqual(get_conj_fn(num1), compiled_get_conj_fn(num1))

    @unittest.skipUnless(_plat_bits == 64, 'Complex return values not '
                         'supported on 32-bit systems.')
    def test_get_complex_constant_fn (self):
        compiled_get_complex_constant_fn = self.jit(
            argtypes = [], restype = complex128)(get_complex_constant_fn)
        self.assertEqual(get_complex_constant_fn(),
                         compiled_get_complex_constant_fn())

    @unittest.skipUnless(_plat_bits == 64, 'Complex return values not '
                         'supported on 32-bit systems.')
    def test_prod_sum_fn (self):
        compiled_prod_sum_fn = self.jit(argtypes = [complex128, complex128, complex128],
                                        restype = complex128)(prod_sum_fn)
        rng = numpy.arange(-1., 1.1, 0.5)
        for ar, ai, xr, xi, br, bi in itertools.product(rng, rng, rng, rng, rng,
                                                        rng):
            a = numpy.complex128(ar + ai * 1j)
            x = numpy.complex128(xr + xi * 1j)
            b = numpy.complex128(br + bi * 1j)
            self.assertEqual(prod_sum_fn(a, x, b),
                             compiled_prod_sum_fn(a, x, b))

# TODO: AST complex support
#class TestASTComplex(test_support.ASTTestCase, TestComplex):
#    pass


# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_complex.py
