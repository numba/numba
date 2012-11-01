#! /usr/bin/env python

from numba import *

import unittest

def unary_minus(x):
    return -x

def unary_not(x):
    return not x

def unary_xor(x):
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

    def test_unary_xor(self):
        test_fn = jit(argtypes=(int_,), restype=int_)(unary_xor)
        test_val = 0xf0f0f0f0
        self.assertEqual(test_fn(test_val), ~test_val)
        self.assertEqual(test_fn(test_val), unary_xor(test_val))


if __name__ == "__main__":
    unittest.main()
