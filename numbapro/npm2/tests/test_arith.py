from __future__ import division
import sys
import numpy as np
from ..compiler import compile
from ..types import (
    int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    float32, float64, complex64, complex128
)
from .support import testcase, main

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

def mod(a, b):
    return a % b

def expr1(a, b, c):
    ac = a * 2
    bc = b * 5
    cc = c * 3
    return (ac + 1) * (bc + 1) * (cc + 1)

def shifts(a, b):
    return (a >> b) << 8

iset = [int8, int16, int32, int64, uint8, uint16, uint32, uint64]
fset = [float32, float64]
cset = [complex64, complex128]

#------------------------------------------------------------------------------
# add

@testcase
def test_add_integer():
    def run(ty, a, b):
        cadd = compile(add, ty, [ty, ty])
        got = cadd(a, b)
        exp = add(a, b)
        assert got == exp, 'add(%s, %s) got = %s expect=%s' % (a, b, got, exp)

    for ty in iset:
        run(ty, 12, 34)


@testcase
def test_add_float():
    def run(ty, a, b):
        cadd = compile(add, ty, [ty, ty])
        got = cadd(a, b)
        exp = add(a, b)
        assert np.allclose(got, exp), 'add(%s, %s) got = %s expect=%s' % (a, b, got, exp)

    for ty in fset:
        run(ty, 1.234, 2.345)


@testcase
def test_add_complex():
    def run(ty, a, b):
        cadd = compile(add, ty, [ty, ty])
        got = cadd(a, b)
        exp = add(a, b)
        assert np.allclose(got, exp), 'add(%s, %s) got = %s expect=%s' % (a, b, got, exp)

    for ty in cset:
        run(ty, 1.2+34j, 2.4+56j)

##------------------------------------------------------------------------------
## sub
#
#@testcase
#def test_sub_integer():
#    def run(ty, a, b):
#        csub = compile(sub, ty, [ty, ty])
#        got = csub(a, b)
#        exp = sub(a, b)
#        assert got == exp, 'sub(%s, %s) got = %s expect=%s' % (a, b, got, exp)
#
#    for ty in iset:
#        run(ty, 45, 12)
#
#
#@testcase
#def test_sub_float():
#    def run(ty, a, b):
#        csub = compile(sub, ty, [ty, ty])
#        got = csub(a, b)
#        exp = sub(a, b)
#        assert np.allclose(got, exp), 'sub(%s, %s) got = %s expect=%s' % (a, b, got, exp)
#
#    for ty in fset:
#        run(ty, 1.234, 2.345)
#
#
#@testcase
#def test_sub_complex():
#    def run(ty, a, b):
#        csub = compile(sub, ty, [ty, ty])
#        got = csub(a, b)
#        exp = sub(a, b)
#        assert np.allclose(got, exp), 'sub(%s, %s) got = %s expect=%s' % (a, b, got, exp)
#
#    for ty in cset:
#        run(ty, 1.2+34j, 2.4+56j)
#
##------------------------------------------------------------------------------
## mul
#
#@testcase
#def test_mul_integer():
#    def run(ty, a, b):
#        cmul = compile(mul, ty, [ty, ty])
#        got = cmul(a, b)
#        exp = mul(a, b)
#        assert got == exp, 'mul(%s, %s) got = %s expect=%s' % (a, b, got, exp)
#
#    for ty in iset:
#        run(ty, 2, 3)
#
#@testcase
#def test_mul_float():
#    def run(ty, a, b):
#        cmul = compile(mul, ty, [ty, ty])
#        got = cmul(a, b)
#        exp = mul(a, b)
#        assert np.allclose(got, exp), 'mul(%s, %s) got = %s expect=%s' % (a, b, got, exp)
#
#    for ty in fset:
#        run(ty, 1.234, 2.345)
#
#
#@testcase
#def test_mul_complex():
#    def run(ty, a, b):
#        cmul = compile(mul, ty, [ty, ty])
#        got = cmul(a, b)
#        exp = mul(a, b)
#        assert np.allclose(got, exp), 'mul(%s, %s) got = %s expect=%s' % (a, b, got, exp)
#
#    for ty in cset:
#        run(ty, 1.2+34j, 2.4+56j)
##------------------------------------------------------------------------------
## div
#
#if sys.platform != 'win32':
#    @testcase
#    def test_div_integer():
#        def run(ty, a, b):
#            cdiv = compile(div, ty, [ty, ty])
#            got = cdiv(a, b)
#            exp = div(a, b)
#            assert got == exp, 'div(%s, %s) got = %s expect=%s' % (a, b, got, exp)
#
#        for ty in iset:
#            run(ty, 4, 2)
#
#@testcase
#def test_div_float():
#    def run(ty, a, b):
#        cdiv = compile(div, ty, [ty, ty])
#        got = cdiv(a, b)
#        exp = div(a, b)
#        assert np.allclose(got, exp), 'div(%s, %s) got = %s expect=%s' % (a, b, got, exp)
#
#    for ty in fset:
#        run(ty, 1.234, 2.345)
#
#
##------------------------------------------------------------------------------
## floordiv
#
#@testcase
#def test_floordiv_integer():
#    def run(ty, a, b):
#        cfloordiv = compile(floordiv, ty, [ty, ty])
#        got = cfloordiv(a, b)
#        exp = floordiv(a, b)
#        assert got == exp, 'floordiv(%s, %s) got = %s expect=%s' % (a, b, got, exp)
#
#    for ty in iset:
#        run(ty, 4, 2)
#
#@testcase
#def test_floordiv_float():
#    def run(ty, a, b):
#        cfloordiv = compile(floordiv, ty, [ty, ty])
#        got = cfloordiv(a, b)
#        exp = floordiv(a, b)
#        assert got == exp, 'floordiv(%s, %s) got = %s expect=%s' % (a, b, got, exp)
#
#    for ty in fset:
#        run(ty, 1.234, 2.345)
#
#
#
##------------------------------------------------------------------------------
## mod
#
#
#@testcase
#def test_mod_integer():
#    def run(ty, a, b):
#        cmod = compile(mod, ty, [ty, ty])
#        got = cmod(a, b)
#        exp = mod(a, b)
#        assert got == exp, 'mod(%s, %s) got = %s expect=%s' % (a, b, got, exp)
#
#    for ty in iset:
#        run(ty, 121, 11)
#
#if sys.platform != 'win32':
#    '''Known problem that llvm generates the wrong symbol for fmodf
#    '''
#    @testcase
#    def test_mod_float():
#        def run(ty, a, b):
#            cmod = compile(mod, ty, [ty, ty])
#            got = cmod(a, b)
#            exp = mod(a, b)
#            assert got == exp, 'mod(%s, %s) got = %s expect=%s' % (a, b, got, exp)
#
#        for ty in fset:
#            run(ty, 432., 21.)
#
#
##------------------------------------------------------------------------------
## expr1
#
#@testcase
#def test_expr1_integer():
#    def run(ty, a, b, c):
#        cexpr1 = compile(expr1, ty, [ty, ty, ty])
#        got = cexpr1(a, b, c)
#        exp = expr1(a, b, c)
#        msg = 'expr1(%s, %s, %s) got = %s expect=%s'
#        assert got == exp, msg % (a, b, c, got, exp)
#
#    for ty in set([int32, int64]):
#        run(ty, 121, 11, 231)
#
##------------------------------------------------------------------------------
## shifts
#
#@testcase
#def test_shifts_signed():
#    cfunc = compile(shifts, int32, [int32, int32])
#
#    a, b = -0xdead, 12
#    got = cfunc(a, b)
#    exp = shifts(a, b)
#    assert got == exp, (got, exp)
#
#@testcase
#def test_shifts_unsigned():
#    cfunc = compile(shifts, uint32, [uint32, uint32])
#
#    a, b = 0xdead, 12
#    got = cfunc(a, b)
#    exp = shifts(a, b)
#    assert got == exp, (got, exp)
#
#

if __name__ == '__main__':
    main()

