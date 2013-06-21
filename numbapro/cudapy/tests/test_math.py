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


def unary_template_float32(func, npfunc, start=0, stop=1):
    unary_template(func, npfunc, np.float32, float32, start, stop)

def unary_template_float64(func, npfunc, start=0, stop=1):
    unary_template(func, npfunc, np.float64, float64, start, stop)

def unary_template(func, npfunc, npdtype, npmtype, start, stop):
    nelem = 100
    A = np.linspace(start, stop, nelem).astype(npdtype)
    B = np.empty_like(A)
    arytype = arraytype(npmtype, 1, 'C')
    cfunc = cudapy.compile_kernel(func, [arytype, arytype])
    cfunc.bind()
    cfunc[1, nelem](A, B)
    assert np.allclose(npfunc(A), B)



def binary_template_float32(func, npfunc, start=0, stop=1):
    binary_template(func, npfunc, np.float32, float32, start, stop)

def binary_template_float64(func, npfunc, start=0, stop=1):
    binary_template(func, npfunc, np.float64, float64, start, stop)

def binary_template(func, npfunc, npdtype, npmtype, start, stop):
    nelem = 100
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




if __name__ == '__main__':
    main()

