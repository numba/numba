from __future__ import print_function, absolute_import, division
import sys
import numpy as np
from numba.cuda.testing import unittest, SerialMixin
from numba import cuda, float32, float64, int32
import math


def math_acos(A, B):
    i = cuda.grid(1)
    B[i] = math.acos(A[i])


def math_asin(A, B):
    i = cuda.grid(1)
    B[i] = math.asin(A[i])


def math_atan(A, B):
    i = cuda.grid(1)
    B[i] = math.atan(A[i])


def math_acosh(A, B):
    i = cuda.grid(1)
    B[i] = math.acosh(A[i])


def math_asinh(A, B):
    i = cuda.grid(1)
    B[i] = math.asinh(A[i])


def math_atanh(A, B):
    i = cuda.grid(1)
    B[i] = math.atanh(A[i])


def math_cos(A, B):
    i = cuda.grid(1)
    B[i] = math.cos(A[i])


def math_sin(A, B):
    i = cuda.grid(1)
    B[i] = math.sin(A[i])


def math_tan(A, B):
    i = cuda.grid(1)
    B[i] = math.tan(A[i])


def math_cosh(A, B):
    i = cuda.grid(1)
    B[i] = math.cosh(A[i])


def math_sinh(A, B):
    i = cuda.grid(1)
    B[i] = math.sinh(A[i])


def math_tanh(A, B):
    i = cuda.grid(1)
    B[i] = math.tanh(A[i])


def math_atan2(A, B, C):
    i = cuda.grid(1)
    C[i] = math.atan2(A[i], B[i])


def math_exp(A, B):
    i = cuda.grid(1)
    B[i] = math.exp(A[i])

def math_erf(A, B):
    i = cuda.grid(1)
    B[i] = math.erf(A[i])

def math_erfc(A, B):
    i = cuda.grid(1)
    B[i] = math.erfc(A[i])

def math_expm1(A, B):
    i = cuda.grid(1)
    B[i] = math.expm1(A[i])

def math_fabs(A, B):
    i = cuda.grid(1)
    B[i] = math.fabs(A[i])

def math_gamma(A, B):
    i = cuda.grid(1)
    B[i] = math.gamma(A[i])

def math_lgamma(A, B):
    i = cuda.grid(1)
    B[i] = math.lgamma(A[i])

def math_log(A, B):
    i = cuda.grid(1)
    B[i] = math.log(A[i])


def math_log10(A, B):
    i = cuda.grid(1)
    B[i] = math.log10(A[i])


def math_log1p(A, B):
    i = cuda.grid(1)
    B[i] = math.log1p(A[i])


def math_sqrt(A, B):
    i = cuda.grid(1)
    B[i] = math.sqrt(A[i])


def math_hypot(A, B, C):
    i = cuda.grid(1)
    C[i] = math.hypot(A[i], B[i])


def math_pow(A, B, C):
    i = cuda.grid(1)
    C[i] = math.pow(A[i], B[i])


def math_ceil(A, B):
    i = cuda.grid(1)
    B[i] = math.ceil(A[i])


def math_floor(A, B):
    i = cuda.grid(1)
    B[i] = math.floor(A[i])


def math_copysign(A, B, C):
    i = cuda.grid(1)
    C[i] = math.copysign(A[i], B[i])


def math_fmod(A, B, C):
    i = cuda.grid(1)
    C[i] = math.fmod(A[i], B[i])


def math_modf(A, B, C):
    i = cuda.grid(1)
    C[i] = math.modf(A[i], B[i])


def math_isnan(A, B):
    i = cuda.grid(1)
    B[i] = math.isnan(A[i])


def math_isinf(A, B):
    i = cuda.grid(1)
    B[i] = math.isinf(A[i])

def math_degrees(A, B):
    i = cuda.grid(1)
    B[i] = math.degrees(A[i])

def math_radians(A, B):
    i = cuda.grid(1)
    B[i] = math.radians(A[i])

def math_pow_binop(A, B, C):
    i = cuda.grid(1)
    C[i] = A[i] ** B[i]


def math_mod_binop(A, B, C):
    i = cuda.grid(1)
    C[i] = A[i] % B[i]


class TestCudaMath(SerialMixin, unittest.TestCase):
    def unary_template_float32(self, func, npfunc, start=0, stop=1):
        self.unary_template(func, npfunc, np.float32, float32, start, stop)


    def unary_template_float64(self, func, npfunc, start=0, stop=1):
        self.unary_template(func, npfunc, np.float64, float64, start, stop)


    def unary_template(self, func, npfunc, npdtype, npmtype, start, stop):
        nelem = 50
        A = np.linspace(start, stop, nelem).astype(npdtype)
        B = np.empty_like(A)
        arytype = npmtype[::1]
        cfunc = cuda.jit((arytype, arytype))(func)
        cfunc[1, nelem](A, B)
        self.assertTrue(np.allclose(npfunc(A), B))

    def unary_bool_template_float32(self, func, npfunc, start=0, stop=1):
        self.unary_template(func, npfunc, np.float32, float32, start, stop)


    def unary_bool_template_float64(self, func, npfunc, start=0, stop=1):
        self.unary_template(func, npfunc, np.float64, float64, start, stop)

    def unary_bool_template(self, func, npfunc, npdtype, npmtype, start, stop):
        nelem = 50
        A = np.linspace(start, stop, nelem).astype(npdtype)
        B = np.empty(A.shape, dtype=np.int32)
        iarytype = npmtype[::1]
        oarytype = int32[::1]
        cfunc = cuda.jit((iarytype, oarytype))(func)
        cfunc[1, nelem](A, B)
        self.assertTrue(np.all(npfunc(A), B))


    def binary_template_float32(self, func, npfunc, start=0, stop=1):
        self.binary_template(func, npfunc, np.float32, float32, start, stop)


    def binary_template_float64(self, func, npfunc, start=0, stop=1):
        self.binary_template(func, npfunc, np.float64, float64, start, stop)


    def binary_template(self, func, npfunc, npdtype, npmtype, start, stop):
        nelem = 50
        A = np.linspace(start, stop, nelem).astype(npdtype)
        B = np.empty_like(A)
        arytype = npmtype[::1]
        cfunc = cuda.jit((arytype, arytype, arytype))(func)
        cfunc.bind()
        cfunc[1, nelem](A, A, B)
        self.assertTrue(np.allclose(npfunc(A, A), B))

    # Test helper for math functions when no ufunc exists
    # and dtype specificity is required.
    def _math_vectorize(self, mathfunc, x):
        ret = np.zeros_like(x)
        for k in range(len(x)):
            ret[k] = mathfunc(x[k])
        return ret

    #------------------------------------------------------------------------------
    # test_math_acos

    def test_math_acos(self):
        self.unary_template_float32(math_acos, np.arccos)
        self.unary_template_float64(math_acos, np.arccos)

    #------------------------------------------------------------------------------
    # test_math_asin


    def test_math_asin(self):
        self.unary_template_float32(math_asin, np.arcsin)
        self.unary_template_float64(math_asin, np.arcsin)

    #------------------------------------------------------------------------------
    # test_math_atan


    def test_math_atan(self):
        self.unary_template_float32(math_atan, np.arctan)
        self.unary_template_float64(math_atan, np.arctan)

    #------------------------------------------------------------------------------
    # test_math_acosh


    def test_math_acosh(self):
        self.unary_template_float32(math_acosh, np.arccosh, start=1, stop=2)
        self.unary_template_float64(math_acosh, np.arccosh, start=1, stop=2)

    #------------------------------------------------------------------------------
    # test_math_asinh


    def test_math_asinh(self):
        self.unary_template_float32(math_asinh, np.arcsinh)
        self.unary_template_float64(math_asinh, np.arcsinh)

    #------------------------------------------------------------------------------
    # test_math_atanh


    def test_math_atanh(self):
        self.unary_template_float32(math_atanh, np.arctanh, start=0, stop=.9)
        self.unary_template_float64(math_atanh, np.arctanh, start=0, stop=.9)


    #------------------------------------------------------------------------------
    # test_math_cos


    def test_math_cos(self):
        self.unary_template_float32(math_cos, np.cos)
        self.unary_template_float64(math_cos, np.cos)

    #------------------------------------------------------------------------------
    # test_math_sin


    def test_math_sin(self):
        self.unary_template_float32(math_sin, np.sin)
        self.unary_template_float64(math_sin, np.sin)

    #------------------------------------------------------------------------------
    # test_math_tan


    def test_math_tan(self):
        self.unary_template_float32(math_tan, np.tan)
        self.unary_template_float64(math_tan, np.tan)

    #------------------------------------------------------------------------------
    # test_math_cosh


    def test_math_cosh(self):
        self.unary_template_float32(math_cosh, np.cosh)
        self.unary_template_float64(math_cosh, np.cosh)

    #------------------------------------------------------------------------------
    # test_math_sinh


    def test_math_sinh(self):
        self.unary_template_float32(math_sinh, np.sinh)
        self.unary_template_float64(math_sinh, np.sinh)

    #------------------------------------------------------------------------------
    # test_math_tanh


    def test_math_tanh(self):
        self.unary_template_float32(math_tanh, np.tanh)
        self.unary_template_float64(math_tanh, np.tanh)

    #------------------------------------------------------------------------------
    # test_math_atan2


    def test_math_atan2(self):
        self.binary_template_float32(math_atan2, np.arctan2)
        self.binary_template_float64(math_atan2, np.arctan2)

    #------------------------------------------------------------------------------
    # test_math_erf


    def test_math_erf(self):
        def ufunc(x):
            return self._math_vectorize(math.erf, x)
        self.unary_template_float32(math_erf, ufunc)
        self.unary_template_float64(math_erf, ufunc)

    #------------------------------------------------------------------------------
    # test_math_erfc


    def test_math_erfc(self):
        def ufunc(x):
            return self._math_vectorize(math.erfc, x)
        self.unary_template_float32(math_erfc, ufunc)
        self.unary_template_float64(math_erfc, ufunc)

    #------------------------------------------------------------------------------
    # test_math_exp


    def test_math_exp(self):
        self.unary_template_float32(math_exp, np.exp)
        self.unary_template_float64(math_exp, np.exp)

    #------------------------------------------------------------------------------
    # test_math_expm1

    def test_math_expm1(self):
        self.unary_template_float32(math_expm1, np.expm1)
        self.unary_template_float64(math_expm1, np.expm1)

    #------------------------------------------------------------------------------
    # test_math_fabs


    def test_math_fabs(self):
        self.unary_template_float32(math_fabs, np.fabs, start=-1)
        self.unary_template_float64(math_fabs, np.fabs, start=-1)

    #------------------------------------------------------------------------------
    # test_math_gamma


    def test_math_gamma(self):
        def ufunc(x):
            return self._math_vectorize(math.gamma, x)
        self.unary_template_float32(math_gamma, ufunc, start=0.1)
        self.unary_template_float64(math_gamma, ufunc, start=0.1)

    #------------------------------------------------------------------------------
    # test_math_lgamma


    def test_math_lgamma(self):
        def ufunc(x):
            return self._math_vectorize(math.lgamma, x)
        self.unary_template_float32(math_lgamma, ufunc, start=0.1)
        self.unary_template_float64(math_lgamma, ufunc, start=0.1)

    #------------------------------------------------------------------------------
    # test_math_log


    def test_math_log(self):
        self.unary_template_float32(math_log, np.log, start=1)
        self.unary_template_float64(math_log, np.log, start=1)

    #------------------------------------------------------------------------------
    # test_math_log10


    def test_math_log10(self):
        self.unary_template_float32(math_log10, np.log10, start=1)
        self.unary_template_float64(math_log10, np.log10, start=1)

    #------------------------------------------------------------------------------
    # test_math_log1p


    def test_math_log1p(self):
        self.unary_template_float32(math_log1p, np.log1p)
        self.unary_template_float64(math_log1p, np.log1p)

    #------------------------------------------------------------------------------
    # test_math_sqrt


    def test_math_sqrt(self):
        self.unary_template_float32(math_sqrt, np.sqrt)
        self.unary_template_float64(math_sqrt, np.sqrt)

    #------------------------------------------------------------------------------
    # test_math_hypot


    def test_math_hypot(self):
        self.binary_template_float32(math_hypot, np.hypot)
        self.binary_template_float64(math_hypot, np.hypot)


    #------------------------------------------------------------------------------
    # test_math_pow


    def test_math_pow(self):
        self.binary_template_float32(math_pow, np.power)
        self.binary_template_float64(math_pow, np.power)


    #------------------------------------------------------------------------------
    # test_math_pow_binop


    def test_math_pow_binop(self):
        self.binary_template_float32(math_pow_binop, np.power)
        self.binary_template_float64(math_pow_binop, np.power)

    #------------------------------------------------------------------------------
    # test_math_ceil


    def test_math_ceil(self):
        self.unary_template_float32(math_ceil, np.ceil)
        self.unary_template_float64(math_ceil, np.ceil)

    #------------------------------------------------------------------------------
    # test_math_floor


    def test_math_floor(self):
        self.unary_template_float32(math_floor, np.floor)
        self.unary_template_float64(math_floor, np.floor)

    #------------------------------------------------------------------------------
    # test_math_copysign


    def test_math_copysign(self):
        self.binary_template_float32(math_copysign, np.copysign, start=-1)
        self.binary_template_float64(math_copysign, np.copysign, start=-1)

    #------------------------------------------------------------------------------
    # test_math_fmod


    def test_math_fmod(self):
        self.binary_template_float32(math_fmod, np.fmod, start=1)
        self.binary_template_float64(math_fmod, np.fmod, start=1)

    #------------------------------------------------------------------------------
    # test_math_mod_binop


    def test_math_mod_binop(self):
        self.binary_template_float32(math_mod_binop, np.fmod, start=1)
        self.binary_template_float64(math_mod_binop, np.fmod, start=1)

    #------------------------------------------------------------------------------
    # test_math_isnan


    def test_math_isnan(self):
        self.unary_bool_template_float32(math_isnan, np.isnan)
        self.unary_bool_template_float64(math_isnan, np.isnan)

    #------------------------------------------------------------------------------
    # test_math_isinf


    def test_math_isinf(self):
        self.unary_bool_template_float32(math_isinf, np.isinf)
        self.unary_bool_template_float64(math_isinf, np.isinf)

    #------------------------------------------------------------------------------
    # test_math_degrees

    def test_math_degrees(self):
        self.unary_bool_template_float32(math_degrees, np.degrees)
        self.unary_bool_template_float64(math_degrees, np.degrees)

    #------------------------------------------------------------------------------
    # test_math_radians

    def test_math_radians(self):
        self.unary_bool_template_float32(math_radians, np.radians)
        self.unary_bool_template_float64(math_radians, np.radians)


if __name__ == '__main__':
    unittest.main()

