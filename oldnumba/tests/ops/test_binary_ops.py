from numba import *
from numba.testing import test_support

import numpy

import unittest


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


class TestBinops(unittest.TestCase):

    def _handle_bitwise_op(self, fn, *args):
        test_fn = jit('object_(object_,object_)')(fn)
        for lhs, rhs in [(5, 2), (0xdead, 0xbeef), (-5, 2)]:
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

    def test_stringformat(self):
        self.assertEqual(autojit(mod)("hello %s", "world"), "hello world")

if __name__ == "__main__":
    test_support.main()
