"""
Tests for @cfunc and friends.
"""

from __future__ import print_function, absolute_import

import ctypes

import llvmlite.binding as ll

import numpy as np

from numba import unittest_support as unittest
from numba import cfunc, types, typing, utils
from numba.types.abstract import _typecache
from numba import jit, numpy_support
from .support import TestCase, tag


def add_usecase(a, b):
    return a + b

add_sig = "float64(float64, float64)"

def objmode_usecase(a, b):
    object()
    return a + b


class TestCFunc(TestCase):

    @tag('important')
    def test_basic(self):
        """
        Basic usage and properties of a cfunc.
        """
        f = cfunc(add_sig)(add_usecase)

        self.assertEqual(f.__name__, "add_usecase")
        self.assertEqual(f.__qualname__, "add_usecase")
        self.assertIs(f.__wrapped__, add_usecase)

        symbol = f.native_name
        self.assertIsInstance(symbol, str)
        self.assertIn("add_usecase", symbol)

        addr = f.address
        self.assertIsInstance(addr, utils.INT_TYPES)

        ct = f.ctypes
        self.assertEqual(ctypes.cast(ct, ctypes.c_void_p).value, addr)

        self.assertPreciseEqual(ct(2.0, 3.5), 5.5)

    def test_llvm_ir(self):
        f = cfunc(add_sig)(add_usecase)
        ir = f.inspect_llvm()
        self.assertIn(f.native_name, ir)
        self.assertIn("fadd double", ir)

    def test_object_mode(self):
        """
        Object mode is currently unsupported.
        """
        with self.assertRaises(NotImplementedError):
            cfunc(add_sig, forceobj=True)(add_usecase)
        with self.assertTypingError() as raises:
            cfunc(add_sig)(objmode_usecase)
        self.assertIn("Untyped global name 'object'", str(raises.exception))


if __name__ == "__main__":
    unittest.main()
