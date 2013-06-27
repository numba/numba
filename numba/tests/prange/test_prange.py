# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import numba
from numba import autojit, double

import unittest
# from unittest import FunctionTestCase as testcase

import numpy as np

tests = []
def testcase(f):
    tests.append(f)
    return f

#----------------------------------------------------------------------------
# Simple isolated tests
#----------------------------------------------------------------------------

@testcase
def test_simple_prange_shared():
    @autojit(warn=False)
    def simple_prange_shared():
        """
        >>> simple_prange_shared()
        20L
        """
        result = np.empty(1, dtype=np.int64)
        shared = 20

        for i in numba.prange(1):
            result[0] = shared
        return result[0]
    assert simple_prange_shared() == 20

@testcase
def test_simple_prange_private():
    @autojit(warn=False)
    def simple_prange_private():
        """
        >>> simple_prange_private()
        10L
        """
        result = np.empty(1, dtype=np.int64)
        var = 20
        for i in numba.prange(1):
            var = 10
            result[0] = var
        return result[0]
    assert simple_prange_private() == 10

@testcase
def test_simple_prange_lastprivate():
    @autojit(warn=False)
    def simple_prange_lastprivate():
        """
        >>> simple_prange_lastprivate()
        10
        """
        var = 20
        for i in numba.prange(1):
            var = 10
        return var
    assert simple_prange_lastprivate() == 10
    

@testcase
def test_simple_prange_reduction():
    @autojit(warn=False)
    def simple_prange_reduction():
        """
        >>> simple_prange_reduction()
        15
        """
        var = 10
        for i in numba.prange(1):
            var += 5
        return var
    assert simple_prange_reduction() == 15

#----------------------------------------------------------------------------
# Error Tests
#----------------------------------------------------------------------------

@autojit(warn=False)
def prange_reduction_error():
    """
    DISABLED.

    >> prange_reduction_error()
    Traceback (most recent call last):
        ...
    NumbaError: 32:8: Local variable  'sum' is not bound yet
    """
    for i in numba.prange(10):
        sum += i
    sum = 0.0
    return sum

#----------------------------------------------------------------------------
# Advanced Tests
#----------------------------------------------------------------------------

@testcase
def test_prange_reduction2():
    @autojit(warn=False)
    def prange_reduction2():
        """
        >>> prange_reduction2()
        49999995000000.0
        """
        sum = 0.0
        for i in numba.prange(10000000):
            sum += i
        return sum
    assert prange_reduction2() == 49999995000000.0

@testcase
def test_prange_reduction_and_privates():
    @autojit(warn=False)
    def prange_reduction_and_privates():
        """
        >>> prange_reduction_and_privates()
        100.0
        """
        sum = 10.0
        for i in numba.prange(10):
            j = i * 2
            sum += j
        return sum
    assert prange_reduction_and_privates() == 100.0

@testcase
def test_prange_lastprivate():
    @autojit(warn=False)
    def prange_lastprivate():
        """
        >>> prange_lastprivate()
        100.0
        18
        """
        sum = 10.0
        for i in numba.prange(10):
            j = i * 2
            sum += j
        print(sum)
        return j
    assert prange_lastprivate() == 18

@testcase
def test_prange_shared_privates_reductions():
    @autojit(warn=False)
    def prange_shared_privates_reductions(shared):
        """
        >>> prange_shared_privates_reductions(2.0)
        100.0
        """
        sum = 10.0

        for i in numba.prange(10):
            j = i * shared
            sum += j
        shared = 3.0
        return sum
    assert prange_shared_privates_reductions(2.0) == 100.0

@testcase
def test_test_sum2d():
    @autojit(warn=False)
    def test_sum2d(A):
        """
        >>> a = np.arange(100).reshape(10, 10)
        >>> test_sum2d(a)
        4950.0
        >>> test_sum2d(a.astype(np.complex128))
        (4950+0j)
        >>> np.sum(a)
        4950
        """
        sum = 0.0
        for i in numba.prange(A.shape[0]):
            for j in range(A.shape[1]):
                # print(i, j)
                sum += A[i, j]
        return sum

    a = np.arange(100).reshape(10, 10)
    assert test_sum2d(a) == 4950.0
    assert test_sum2d(a.astype(np.complex128)) == 4950+0j
    assert np.sum(a) == 4950

@testcase
def test_test_prange_in_closure():
    @autojit(warn=False)
    def test_prange_in_closure(x):
        """
        >>> test_prange_in_closure(2.0)()
        1000.0
        """
        sum = 10.0
        N = 10

        @double()
        def inner():
            sum = 100.0
            for i in numba.prange(N):
                for j in range(N):
                    sum += i * x
            return sum
        return inner
    assert test_prange_in_closure(2.0)() == 1000.0

@testcase
def test_test_prange_in_closure2():
    @autojit(warn=False)
    def test_prange_in_closure2(x):
        """
        >>> test_prange_in_closure2(2.0)()
        10000.0
        """
        sum = 10.0
        N = 10

        @double()
        def inner():
            sum = 100.0
            for i in numba.prange(N):
                for j in range(N):
                    sum += (i * N + j) * x
            return sum
        return inner

    assert test_prange_in_closure2(2.0)() == 10000.0

if __name__ == '__main__':
    # unittest.main()
    for test in tests:
        print("running", test.__name__)
        test()