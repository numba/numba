#! /usr/bin/env python
# ______________________________________________________________________
'''test_complex

Test Numba's ability to generate code that supports complex numbers.
'''
# ______________________________________________________________________

import cmath
import unittest
import numba
from numba import *
from numba.decorators import jit
from numba.utils import debugout
from numba.llvm_types import _plat_bits
from numba.testing import test_support

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

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    return a / b

def floordiv(a, b):
    return a // b

def sqrt(a, b):
    result = a**2 + b**2
    return cmath.sqrt(result) + 1.6j

def log(a, b):
    result = a**2 + b**2
    return cmath.log(result) + 1.6j

def log10(a, b):
    result = a**2 + b**2
    return cmath.log10(result) + 1.6j

def exp(a, b):
    result = a**2 + b**2
    return cmath.exp(result) + 1.6j

def sin(a, b):
    result = a**2 + b**2
    return cmath.sin(result) + 1.6j

def cos(a, b):
    result = a**2 + b**2
    return cmath.cos(result) + 1.6j

def atan(a, b):
    result = a**2 + b**2
    return cmath.atan(result) + 1.6j

def asinh(a, b):
    result = a**2 + b**2
    return cmath.asinh(result) + 1.6j

def cosh(a, b):
    result = a**2 + b**2
    return cmath.cosh(result) + 1.6j

def absolute(a, b):
    result = a**2 + b**2
    return abs(result) + 1.6j

def mandel(x, y, max_iters):
    i = 0
    z = 0.0j
    for i in range(max_iters):
        z = z**2 + (x + y*1j)
        if abs(z**2) >= 4:
            return i

    return 255

# ______________________________________________________________________

m, n = 0.4 + 1.2j, 5.1 - 0.6j

class TestASTComplex(test_support.ASTTestCase):

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

    def test_get_conj_fn (self):
        num0 = 4 - 1.5j
        num1 = numpy.complex128(num0)
        compiled_get_conj_fn = self.jit(argtypes = [complex128],
                                        restype = complex128)(get_conj_fn)
        self.assertEqual(compiled_get_conj_fn(num0), 4 + 1.5j)
        self.assertEqual(get_conj_fn(num0), compiled_get_conj_fn(num0))
        self.assertEqual(compiled_get_conj_fn(num1), 4 + 1.5j)
        self.assertEqual(get_conj_fn(num1), compiled_get_conj_fn(num1))

    def test_get_complex_constant_fn (self):
        compiled_get_complex_constant_fn = self.jit(
            argtypes = [], restype = complex128)(get_complex_constant_fn)
        self.assertEqual(get_complex_constant_fn(),
                         compiled_get_complex_constant_fn())

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

    def test_arithmetic_mixed(self):
        m, n = 0.4 + 1.2j, 10.0
        self.arithmetic(m, n)
        m, n = 0.4 + 1.2j, 10
        self.arithmetic(m, n)

    def arithmetic(self, m, n):
        self.assertAlmostEqual(self.autojit(add)(m, n), add(m, n))
        self.assertAlmostEqual(self.autojit(sub)(m, n), sub(m, n))
        self.assertAlmostEqual(self.autojit(mul)(m, n), mul(m, n))
        self.assertAlmostEqual(self.autojit(div)(m, n), div(m, n))
        if not numba.PY3:
            self.assertAlmostEqual(self.autojit(floordiv)(m, n), floordiv(m, n))

    def test_complex_math(self):
        self.assertAlmostEqual(self.autojit(sqrt)(m, n), sqrt(m, n))
        self.assertAlmostEqual(self.autojit(log)(m, n), log(m, n))
        self.assertAlmostEqual(self.autojit(log10)(m, n), log10(m, n))
        self.assertAlmostEqual(self.autojit(exp)(m, n), exp(m, n), places=3)
        self.assertAlmostEqual(self.autojit(sin)(m, n), sin(m, n))
        self.assertAlmostEqual(self.autojit(cos)(m, n), cos(m, n))
        self.assertAlmostEqual(self.autojit(cosh)(m, n), cosh(m, n), places=3)
        self.assertAlmostEqual(self.autojit(atan)(m, n), atan(m, n))
        self.assertAlmostEqual(self.autojit(asinh)(m, n), asinh(m, n))
        self.assertAlmostEqual(self.autojit(absolute)(m, n), absolute(m, n))


    def test_complex_math_float_input(self):
        m, n = .12, .32
        self.assertAlmostEqual(self.autojit(sqrt)(m, n), sqrt(m, n))
        self.assertAlmostEqual(self.autojit(log)(m, n), log(m, n))
        self.assertAlmostEqual(self.autojit(log10)(m, n), log10(m, n))
        self.assertAlmostEqual(self.autojit(exp)(m, n), exp(m, n))
        self.assertAlmostEqual(self.autojit(sin)(m, n), sin(m, n))
        self.assertAlmostEqual(self.autojit(cos)(m, n), cos(m, n))
        self.assertAlmostEqual(self.autojit(cosh)(m, n), cosh(m, n))
        self.assertAlmostEqual(self.autojit(atan)(m, n), atan(m, n))
        self.assertAlmostEqual(self.autojit(asinh)(m, n), asinh(m, n))
        self.assertAlmostEqual(self.autojit(absolute)(m, n), absolute(m, n))


    def test_mandel(self):
        self.assertEqual(self.autojit(mandel)(-1, -1, 20), 2)
        self.assertEqual(mandel(-1, -1, 20), 2)

# ______________________________________________________________________

if __name__ == "__main__":
    # m, n = .12, .32
    # print autojit(add)(m, n)
    # print autojit(cosh)(m, n), cosh(m, n)
    # print(autojit(sqrt)(m, n))
    # num0 = 0 - 2j
    # num1 = numpy.complex128(num0)
    # compiled_get_imag_fn = jit(argtypes = [complex128])(get_imag_fn)
    # compiled_get_imag_fn(num0)
    unittest.main() #verbosity=3)

# ______________________________________________________________________
# End of test_complex.py
