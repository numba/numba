# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numba import *
from numba.testing import test_support

import numpy

import unittest

# NOTE: See also numba.tests.ops.test_binary_ops

def maxstar1d(a, b):
    M = a.shape[0]
    res = numpy.empty(M)
    for i in range(M):
        res[i] = numpy.max(a[i], b[i]) + numpy.log1p(
            numpy.exp(-numpy.abs(a[i] - b[i])))
    return res

class TestIssue56(unittest.TestCase):
    def test_maxstar1d(self):
        test_fn = jit('f8[:](f8[:],f8[:])')(maxstar1d)
        test_a = numpy.random.random(10)
        test_b = numpy.random.random(10)
        self.assertTrue(numpy.allclose(test_fn(test_a, test_b),
                                       maxstar1d(test_a, test_b)))

if __name__ == "__main__":
#    TestIssue56("test_maxstar1d").debug()
    test_support.main()
