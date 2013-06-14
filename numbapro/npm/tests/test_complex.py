import numpy as np
from ..compiler import compile
from ..types import *
from .support import testcase, main

def complex_real(c):
    return c.real

def complex_imag(c):
    return c.imag

def complex_arith(c, d):
    return complex(c.real * d.imag, c.imag * d.real)

@testcase
def test_complex_real():
    c = 123+234j
    compiled = compile(complex_real, float32, [complex64])
    got = compiled(c)
    exp = complex_real(c)
    assert np.allclose(got, exp), (got, exp)

@testcase
def test_complex_imag():
    c = 123+234j
    compiled = compile(complex_imag, float32, [complex64])
    got = compiled(c)
    exp = complex_imag(c)
    assert np.allclose(got, exp), (got, exp)

@testcase
def test_complex_arith():
    c, d = 123+234j, 321-432j
    compiled = compile(complex_arith, complex64, [complex64, complex64])
    exp = complex_arith(c, d)
    got = compiled(c, d)
    assert np.allclose(got, exp), (got, exp)

if __name__ == '__main__':
    main()
