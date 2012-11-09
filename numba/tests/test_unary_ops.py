#! /usr/bin/env python

from numba import *
from . import test_support

import numpy, math

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

def maxstar1d(a, b):
    M = a.shape[0]
    res = numpy.empty(M)
    for i in range(M):
        res[i] = numpy.max(a[i], b[i]) + numpy.log1p(
            numpy.exp(-numpy.abs(a[i] - b[i])))
    return res


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

    @test_support.checkSkipFlag("Object arithmetic not currently supported.")
    def test_maxstar1d(self):
        test_fn = jit('f8[:](f8[:],f8[:])')(maxstar1d)
        test_a = numpy.random.random(10)
        test_b = numpy.random.random(10)
        self.assertTrue((test_fn(test_a, test_b) ==
                         maxstar1d(test_a, test_b)).all())


if __name__ == "__main__":
    test_support.main()
