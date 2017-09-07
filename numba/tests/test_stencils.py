#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

from __future__ import print_function, division, absolute_import

import sys
import numpy as np

import numba
from numba import unittest_support as unittest
from numba import njit, stencil
from .support import tag

# for decorating tests, marking that Windows with Python 2.7 is not supported
_windows_py27 = (sys.platform.startswith('win32') and
                 sys.version_info[:2] == (2, 7))
_32bit = sys.maxsize <= 2 ** 32
_reason = 'parfors not supported'
skip_unsupported = unittest.skipIf(_32bit or _windows_py27, _reason)


class TestStencils(unittest.TestCase):

    def __init__(self, *args):
        super(TestStencils, self).__init__(*args)

    @skip_unsupported
    @tag('important')
    def test_stencil1(self):
        @stencil()
        def stencil1_kernel(a):
            return 0.25 * (a[0,1] + a[1,0] + a[0,-1] + a[-1,0])

        def test_with_out(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.zeros(n**2).reshape((n, n))
            B = stencil1_kernel(A, out=B)
            return B

        test_with_out_njit = njit(test_with_out)
        test_with_out_par = njit(test_with_out, parallel=True)


        def test_without_out(n):
            A = np.arange(n**2).reshape((n, n))
            B = stencil1_kernel(A)
            return B

        test_without_out_njit = njit(test_without_out)
        test_without_out_par = njit(test_without_out, parallel=True)

        def test_impl_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.zeros(n**2).reshape((n, n))
            for i in range(1, n-1):
                for j in range(1, n-1):
                    B[i,j] = 0.25 * (A[i,j+1] + A[i+1,j] + A[i,j-1] + A[i-1,j])
            return B

        n = 100
        out_seq = test_with_out(n)
        out_njit = test_with_out_njit(n)
        out_par = test_with_out_par(n)
        with_seq = test_without_out(n)
        with_njit = test_without_out_njit(n)
        with_par = test_without_out_par(n)
        py_output = test_impl_seq(n)

        np.testing.assert_almost_equal(out_seq, py_output, decimal=1)
        np.testing.assert_almost_equal(out_njit, py_output, decimal=1)
        np.testing.assert_almost_equal(out_par, py_output, decimal=1)
        np.testing.assert_almost_equal(with_seq, py_output, decimal=1)
        np.testing.assert_almost_equal(with_njit, py_output, decimal=1)
        np.testing.assert_almost_equal(with_par, py_output, decimal=1)

    @skip_unsupported
    @tag('important')
    def test_stencil2(self):
        @stencil(neighborhood=((-5, 0), ))
        def stencil2_kernel(a):
            cum = a[-5]
            for i in range(-4,1):
                cum += a[i]
            return 0.3 * cum

        def test_seq(n):
            A = np.arange(n)
            B = stencil2_kernel(A)
            return B

        test_njit = njit(test_seq)
        test_par = njit(test_seq, parallel=True)

        def test_impl_seq(n):
            A = np.arange(n)
            B = np.zeros(n)
            for i in range(5, len(A)):
                B[i] = 0.3 * sum(A[i-5:i+1])
            return B

        n = 100
        seq_res = test_seq(n)
        njit_res = test_njit(n)
        par_res = test_par(n)
        py_output = test_impl_seq(n)

        np.testing.assert_almost_equal(seq_res, py_output, decimal=1)
        np.testing.assert_almost_equal(njit_res, py_output, decimal=1)
        np.testing.assert_almost_equal(par_res, py_output, decimal=1)

if __name__ == "__main__":
    unittest.main()
