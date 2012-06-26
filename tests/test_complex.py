#! /usr/bin/env python
# ______________________________________________________________________
'''test_complex

Test Numba's ability to generate code that supports complex numbers.
'''
# ______________________________________________________________________

import unittest

from numba.decorators import numba_compile
from numba.utils import debugout

import numpy

# ______________________________________________________________________

def get_real_fn (in_num):
    return in_num.real

def get_imag_fn (in_num):
    ret_val = in_num.imag
    debugout("get_imag_fn() in_num.real = ", in_num.real, " in_num.imag = ",
             ret_val)
    return ret_val

def get_conj_fn (in_num):
    return in_num.conjugate()

def get_complex_constant_fn ():
    return (3. + 4.j).conjugate()

# ______________________________________________________________________

class TestComplex (unittest.TestCase):
    def test_get_real_fn (self):
        num0 = 3 + 2j
        num1 = numpy.complex128(num0)
        compiled_get_real_fn = numba_compile(arg_types = ['D'])(get_real_fn)
        self.assertEqual(compiled_get_real_fn(num0), 3.)
        self.assertEqual(get_real_fn(num0), compiled_get_real_fn(num0))
        self.assertEqual(compiled_get_real_fn(num1), 3.)
        self.assertEqual(get_real_fn(num1), compiled_get_real_fn(num1))

    def test_get_imag_fn (self):
        num0 = 0 - 2j
        print num0
        num1 = numpy.complex128(num0)
        print num1
        compiled_get_imag_fn = numba_compile(arg_types = ['D'])(get_imag_fn)
        self.assertEqual(compiled_get_imag_fn(num0), -2.)
        self.assertEqual(get_imag_fn(num0), compiled_get_imag_fn(num0))
        self.assertEqual(compiled_get_imag_fn(num1), -2.)
        self.assertEqual(get_imag_fn(num1), compiled_get_imag_fn(num1))

    def test_get_conj_fn (self):
        num0 = 4 - 1.5j
        num1 = numpy.complex128(num0)
        compiled_get_conj_fn = numba_compile(arg_types = ['D'],
                                             ret_type = 'D')(get_conj_fn)
        self.assertEqual(compiled_get_conj_fn(num0), 4 + 1.5j)
        self.assertEqual(get_conj_fn(num0), compiled_get_conj_fn(num0))
        self.assertEqual(compiled_get_conj_fn(num1), 4 + 1.5j)
        self.assertEqual(get_conj_fn(num1), compiled_get_conj_fn(num1))

    def test_get_complex_constant_fn (self):
        compiled_get_complex_constant_fn = numba_compile(
            arg_types = [], ret_type = 'D')(get_complex_constant_fn)
        self.assertEqual(get_complex_constant_fn(),
                         compiled_get_complex_constant_fn())

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_complex.py
