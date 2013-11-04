import numpy as np
from ..compiler import compile
from ..types import float32, complex64
from .support import testcase, main, assertTrue

def complex_real(c):
    return c.real

def complex_imag(c):
    return c.imag

def complex_arith(c, d):
    return complex(c.real * d.imag, c.imag * d.real)

def complex_ctor_1(a):
    return complex(a)

def complex_ctor_2(a):
    return complex(0, a)

@testcase
def test_complex_real():
    c = 123+234j
    compiled = compile(complex_real, float32, [complex64])
    got = compiled(c)
    exp = complex_real(c)
    assertTrue(np.allclose(got, exp), (got, exp))

@testcase
def test_complex_imag():
    c = 123+234j
    compiled = compile(complex_imag, float32, [complex64])
    got = compiled(c)
    exp = complex_imag(c)
    assertTrue(np.allclose(got, exp), (got, exp))

@testcase
def test_complex_arith():
    c, d = 123+234j, 321-432j
    compiled = compile(complex_arith, complex64, [complex64, complex64])
    exp = complex_arith(c, d)
    got = compiled(c, d)
    assertTrue(np.allclose(got, exp), (got, exp))

@testcase
def test_complex_ctor_1():
    a = 12.3
    compiled = compile(complex_ctor_1, complex64, [float32])
    exp = complex_ctor_1(a)
    got = compiled(a)
    assertTrue(np.allclose(got, exp), (got, exp))


@testcase
def test_complex_ctor_2():
    a = 12.3
    compiled = compile(complex_ctor_2, complex64, [float32])
    exp = complex_ctor_2(a)
    got = compiled(a)
    assertTrue(np.allclose(got, exp), (got, exp))

if __name__ == '__main__':
    main()
