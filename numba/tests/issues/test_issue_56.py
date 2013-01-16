from numba import *
from numba.tests import test_support

import numpy

import unittest


def maxstar1d(a, b):
    M = a.shape[0]
    res = numpy.empty(M)
    for i in range(M):
        res[i] = numpy.max(a[i], b[i]) + numpy.log1p(
            numpy.exp(-numpy.abs(a[i] - b[i])))
    return res

def add(a, b): return a + b

def sub(a, b): return a - b

def mult(a, b): return a * b

def div(a, b): return a / b

def mod(a, b): return a % b

def pow_(a, b): return a ** b

def shl(a, b): return a << b

def shr(a, b): return a >> b

def bitor(a, b): return a | b

def bitxor(a, b): return a ^ b

def bitand(a, b): return a & b

def floor(a, b): return a // b


class TestIssue56(unittest.TestCase):
    def test_maxstar1d(self):
        test_fn = jit('f8[:](f8[:],f8[:])')(maxstar1d)
        test_a = numpy.random.random(10)
        test_b = numpy.random.random(10)
        self.assertTrue((test_fn(test_a, test_b) ==
                         maxstar1d(test_a, test_b)).all())

    def _handle_bitwise_op(self, fn, *args):
        test_fn = jit('object_(object_,object_)')(fn)
        for lhs, rhs in [(5, 2), (0xdead, 0xbeef)]:
            self.assertEquals(fn(lhs, rhs), test_fn(lhs, rhs))
        for lhs, rhs in args:
            self.assertEquals(fn(lhs, rhs), test_fn(lhs, rhs))

    def _handle_binop(self, fn, *args):
        return self._handle_bitwise_op(fn, (6.3, 9.2), *args)

    def test_add(self): self._handle_binop(add, ('egg', 'spam'))
    def test_sub(self): self._handle_binop(sub)
    def test_mult(self): self._handle_binop(mult)
    def test_div(self): self._handle_binop(div)
    def test_mod(self): self._handle_binop(mod)
    def test_pow(self): self._handle_binop(pow_)

    def test_shl(self): self._handle_bitwise_op(shl)
    def test_shr(self): self._handle_bitwise_op(shr)
    def test_bitor(self): self._handle_bitwise_op(bitor)
    def test_bitxor(self): self._handle_bitwise_op(bitxor)
    def test_bitand(self): self._handle_bitwise_op(bitand)

    def test_floor(self): self._handle_binop(floor)


if __name__ == "__main__":
    TestIssue56("test_maxstar1d").debug()
#    test_support.main()
