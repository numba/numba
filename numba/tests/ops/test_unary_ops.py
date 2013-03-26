#! /usr/bin/env python

from numba import *
from numba.testing import test_support

import unittest


def unary_minus(x):
    return -x

def unary_not(x):
    return not x

def unary_not_pred(p):
    if not p:
        return 1
    return 0

def unary_invert(x):
    return ~x


class TestUnaryOps(unittest.TestCase):
    def test_unary_minus(self):
        test_fn = jit(argtypes=(double,), restype=double)(unary_minus)
        test_val = 3.1415
        self.assertEqual(test_fn(test_val), -test_val)
        self.assertEqual(test_fn(test_val), unary_minus(test_val))

    def test_unary_not(self):
        test_fn = jit(argtypes=(bool_,), restype=bool_)(unary_not)
        for test_val in True, False:
            self.assertEqual(test_fn(test_val), not test_val)
            self.assertEqual(test_fn(test_val), unary_not(test_val))

    def test_unary_not_pred(self):
        test_fn = jit(argtypes=(bool_,), restype=int_)(unary_not_pred)
        for test_val in True, False:
            self.assertEqual(test_fn(test_val), 0 if test_val else 1)
            self.assertEqual(test_fn(test_val), unary_not(test_val))

    def test_unary_invert(self):
        test_fn = jit(argtypes=(int_,), restype=int_)(unary_invert)
        test_val = 0x70f0f0f0
        self.assertEqual(test_fn(test_val), ~test_val)
        self.assertEqual(test_fn(test_val), unary_invert(test_val))


if __name__ == "__main__":
    test_support.main()
