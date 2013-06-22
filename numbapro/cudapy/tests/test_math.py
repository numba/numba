import sys
import numpy as np
from .support import testcase, main
from numbapro import cuda
from numbapro import cudapy
from numbapro.npm.types import *
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

def math_expm1(A, B):
    i = cuda.grid(1)
    B[i] = math.expm1(A[i])

def math_fabs(A, B):
    i = cuda.grid(1)
    B[i] = math.fabs(A[i])

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

def math_pow_binop(A, B, C):
    i = cuda.grid(1)
    C[i] = A[i] ** B[i]

def math_mod_binop(A, B, C):
    i = cuda.grid(1)
    C[i] = A[i] % B[i]

def unary_template_float32(func, npfunc, start=0, stop=1):
    unary_template(func, npfunc, np.float32, float32, start, stop)

def unary_template_float64(func, npfunc, start=0, stop=1):
    unary_template(func, npfunc, np.float64, float64, start, stop)

def unary_template(func, npfunc, npdtype, npmtype, start, stop):
    nelem = 50
    A = np.linspace(start, stop, nelem).astype(npdtype)
    B = np.empty_like(A)
    arytype = arraytype(npmtype, 1, 'C')
    cfunc = cudapy.compile_kernel(func, [arytype, arytype])
    cfunc.bind()
    cfunc[1, nelem](A, B)
    assert np.allclose(npfunc(A), B)

def unary_bool_template_float32(func, npfunc, start=0, stop=1):
    unary_template(func, npfunc, np.float32, float32, start, stop)

def unary_bool_template_float64(func, npfunc, start=0, stop=1):
    unary_template(func, npfunc, np.float64, float64, start, stop)

def unary_bool_template(func, npfunc, npdtype, npmtype, start, stop):
    nelem = 50
    A = np.linspace(start, stop, nelem).astype(npdtype)
    B = np.empty(A.shape, dtype=np.int32)
    iarytype = arraytype(A.dtype, 1, 'C')
    oarytype = arraytype(B.dtype, 1, 'C')
    cfunc = cudapy.compile_kernel(func, [iarytype, oarytype])
    cfunc.bind()
    cfunc[1, nelem](A, B)
    assert np.all(npfunc(A), B)



def binary_template_float32(func, npfunc, start=0, stop=1):
    binary_template(func, npfunc, np.float32, float32, start, stop)

def binary_template_float64(func, npfunc, start=0, stop=1):
    binary_template(func, npfunc, np.float64, float64, start, stop)

def binary_template(func, npfunc, npdtype, npmtype, start, stop):
    nelem = 50
    A = np.linspace(start, stop, nelem).astype(npdtype)
    B = np.empty_like(A)
    arytype = arraytype(npmtype, 1, 'C')
    cfunc = cudapy.compile_kernel(func, [arytype, arytype, arytype])
    cfunc.bind()
    cfunc[1, nelem](A, A, B)
    assert np.allclose(npfunc(A, A), B)


#------------------------------------------------------------------------------
# test_math_acos

@testcase
def test_math_acos():
    unary_template_float32(math_acos, np.arccos)
    unary_template_float64(math_acos, np.arccos)

#------------------------------------------------------------------------------
# test_math_asin

@testcase
def test_math_asin():
    unary_template_float32(math_asin, np.arcsin)
    unary_template_float64(math_asin, np.arcsin)

#------------------------------------------------------------------------------
# test_math_atan

@testcase
def test_math_atan():
    unary_template_float32(math_atan, np.arctan)
    unary_template_float64(math_atan, np.arctan)

#------------------------------------------------------------------------------
# test_math_acosh

@testcase
def test_math_acosh():
    unary_template_float32(math_acosh, np.arccosh, start=1, stop=2)
    unary_template_float64(math_acosh, np.arccosh, start=1, stop=2)

#------------------------------------------------------------------------------
# test_math_asinh

@testcase
def test_math_asinh():
    unary_template_float32(math_asinh, np.arcsinh)
    unary_template_float64(math_asinh, np.arcsinh)

#------------------------------------------------------------------------------
# test_math_atanh

@testcase
def test_math_atanh():
    unary_template_float32(math_atanh, np.arctanh)
    unary_template_float64(math_atanh, np.arctanh)



#------------------------------------------------------------------------------
# test_math_cos

@testcase
def test_math_cos():
    unary_template_float32(math_cos, np.cos)
    unary_template_float64(math_cos, np.cos)

#------------------------------------------------------------------------------
# test_math_sin

@testcase
def test_math_sin():
    unary_template_float32(math_sin, np.sin)
    unary_template_float64(math_sin, np.sin)

#------------------------------------------------------------------------------
# test_math_tan

@testcase
def test_math_tan():
    unary_template_float32(math_tan, np.tan)
    unary_template_float64(math_tan, np.tan)

#------------------------------------------------------------------------------
# test_math_cosh

@testcase
def test_math_cosh():
    unary_template_float32(math_cosh, np.cosh)
    unary_template_float64(math_cosh, np.cosh)

#------------------------------------------------------------------------------
# test_math_sinh

@testcase
def test_math_sinh():
    unary_template_float32(math_sinh, np.sinh)
    unary_template_float64(math_sinh, np.sinh)

#------------------------------------------------------------------------------
# test_math_tanh

@testcase
def test_math_tanh():
    unary_template_float32(math_tanh, np.tanh)
    unary_template_float64(math_tanh, np.tanh)

#------------------------------------------------------------------------------
# test_math_atan2

@testcase
def test_math_atan2():
    binary_template_float32(math_atan2, np.arctan2)
    binary_template_float64(math_atan2, np.arctan2)



#------------------------------------------------------------------------------
# test_math_exp

@testcase
def test_math_exp():
    unary_template_float32(math_exp, np.exp)
    unary_template_float64(math_exp, np.exp)

#------------------------------------------------------------------------------
# test_math_expm1

if sys.version_info[:2] >= (2, 7):
    @testcase
    def test_math_expm1():
        unary_template_float32(math_expm1, np.expm1)
        unary_template_float64(math_expm1, np.expm1)

#------------------------------------------------------------------------------
# test_math_fabs

@testcase
def test_math_fabs():
    unary_template_float32(math_fabs, np.fabs, start=-1)
    unary_template_float64(math_fabs, np.fabs, start=-1)


#------------------------------------------------------------------------------
# test_math_log

@testcase
def test_math_log():
    unary_template_float32(math_log, np.log, start=1)
    unary_template_float64(math_log, np.log, start=1)

#------------------------------------------------------------------------------
# test_math_log10

@testcase
def test_math_log10():
    unary_template_float32(math_log10, np.log10, start=1)
    unary_template_float64(math_log10, np.log10, start=1)

#------------------------------------------------------------------------------
# test_math_log1p

@testcase
def test_math_log1p():
    unary_template_float32(math_log1p, np.log1p)
    unary_template_float64(math_log1p, np.log1p)

#------------------------------------------------------------------------------
# test_math_sqrt

@testcase
def test_math_sqrt():
    unary_template_float32(math_sqrt, np.sqrt)
    unary_template_float64(math_sqrt, np.sqrt)

#------------------------------------------------------------------------------
# test_math_pow

@testcase
def test_math_pow():
    binary_template_float32(math_pow, np.power)
    binary_template_float64(math_pow, np.power)


#------------------------------------------------------------------------------
# test_math_pow_binop

@testcase
def test_math_pow_binop():
    binary_template_float32(math_pow_binop, np.power)
    binary_template_float64(math_pow_binop, np.power)

#------------------------------------------------------------------------------
# test_math_ceil

@testcase
def test_math_ceil():
    unary_template_float32(math_ceil, np.ceil)
    unary_template_float64(math_ceil, np.ceil)

#------------------------------------------------------------------------------
# test_math_floor

@testcase
def test_math_floor():
    unary_template_float32(math_floor, np.floor)
    unary_template_float64(math_floor, np.floor)

#------------------------------------------------------------------------------
# test_math_copysign

@testcase
def test_math_copysign():
    binary_template_float32(math_copysign, np.copysign, start=-1)
    binary_template_float64(math_copysign, np.copysign, start=-1)

#------------------------------------------------------------------------------
# test_math_fmod

@testcase
def test_math_fmod():
    binary_template_float32(math_fmod, np.fmod, start=1)
    binary_template_float64(math_fmod, np.fmod, start=1)

#------------------------------------------------------------------------------
# test_math_mod_binop

@testcase
def test_math_mod_binop():
    binary_template_float32(math_mod_binop, np.fmod, start=1)
    binary_template_float64(math_mod_binop, np.fmod, start=1)

#------------------------------------------------------------------------------
# test_math_isnan

@testcase
def test_math_isnan():
    unary_bool_template_float32(math_isnan, np.isnan)
    unary_bool_template_float64(math_isnan, np.isnan)

#------------------------------------------------------------------------------
# test_math_isinf

@testcase
def test_math_isinf():
    unary_bool_template_float32(math_isinf, np.isinf)
    unary_bool_template_float64(math_isinf, np.isinf)



if __name__ == '__main__':
    main()

