#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

from __future__ import print_function, division, absolute_import

from math import sqrt
import re
import sys
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn

import numba
from numba import unittest_support as unittest
from .support import TestCase
from numba import njit, prange, stencil, inline_closurecall
from numba import compiler, typing
from numba.targets import cpu
from numba import types
from numba.targets.registry import cpu_target
from numba import config
from numba.annotations import type_annotations
from numba.ir_utils import (copy_propagate, apply_copy_propagate,
                            get_name_var_table, remove_dels, remove_dead)
from numba import ir
from numba.compiler import compile_isolated, Flags
from numba.bytecode import ByteCodeIter
from .support import tag
from .matmul_usecase import needs_blas
from .test_linalg import needs_lapack

# for decorating tests, marking that Windows with Python 2.7 is not supported
_windows_py27 = (sys.platform.startswith('win32') and
                 sys.version_info[:2] == (2, 7))
_32bit = sys.maxsize <= 2 ** 32
_reason = 'parfors not supported'
skip_unsupported = unittest.skipIf(_32bit or _windows_py27, _reason)


class TestParforsBase(TestCase):
    """
    Base class for testing parfors.
    Provides functions for compilation and three way comparison between
    python functions, njit'd functions and parfor njit'd functions.
    """

    def __init__(self, *args):
        # flags for njit()
        self.cflags = Flags()
        self.cflags.set('nrt')

        # flags for njit(parallel=True)
        self.pflags = Flags()
        self.pflags.set('auto_parallel', cpu.ParallelOptions(True))
        self.pflags.set('nrt')
        super(TestParforsBase, self).__init__(*args)

    def _compile_this(self, func, sig, flags):
        return compile_isolated(func, sig, flags=flags)

    def compile_parallel(self, func, sig):
        return self._compile_this(func, sig, flags=self.pflags)

    def compile_njit(self, func, sig):
        return self._compile_this(func, sig, flags=self.cflags)

    def compile_all(self, pyfunc, *args, **kwargs):
        sig = tuple([numba.typeof(x) for x in args])

        # compile the prange injected function
        cpfunc = self.compile_parallel(pyfunc, sig)

        # compile a standard njit of the original function
        cfunc = self.compile_njit(pyfunc, sig)

        return cfunc, cpfunc

    def check_prange_vs_others(self, pyfunc, cfunc, cpfunc, *args, **kwargs):
        """
        Checks python, njit and parfor impls produce the same result.

        Arguments:
            pyfunc - the python function to test
            cfunc - CompilerResult from njit of pyfunc
            cpfunc - CompilerResult from njit(parallel=True) of pyfunc
            args - arguments for the function being tested
            kwargs - to pass to np.testing.assert_almost_equal
                     'decimal' is supported.
        """

        # python result
        py_expected = pyfunc(*args)

        # njit result
        njit_output = cfunc.entry_point(*args)

        # parfor result
        parfor_output = cpfunc.entry_point(*args)

        np.testing.assert_almost_equal(njit_output, py_expected, **kwargs)
        np.testing.assert_almost_equal(parfor_output, py_expected, **kwargs)

        # make sure parfor set up scheduling
        self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())


def test1(sptprice, strike, rate, volatility, timev):
    # blackscholes example
    logterm = np.log(sptprice / strike)
    powterm = 0.5 * volatility * volatility
    den = volatility * np.sqrt(timev)
    d1 = (((rate + powterm) * timev) + logterm) / den
    d2 = d1 - den
    NofXd1 = 0.5 + 0.5 * 2.0 * d1
    NofXd2 = 0.5 + 0.5 * 2.0 * d2
    futureValue = strike * np.exp(- rate * timev)
    c1 = futureValue * NofXd2
    call = sptprice * NofXd1 - c1
    put = call - futureValue + sptprice
    return put


def test2(Y, X, w, iterations):
    # logistic regression example
    for i in range(iterations):
        w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y), X)
    return w

def test_kmeans_example(A, numCenter, numIter, init_centroids):
    centroids = init_centroids
    N, D = A.shape

    for l in range(numIter):
        dist = np.array([[sqrt(np.sum((A[i,:]-centroids[j,:])**2))
                                for j in range(numCenter)] for i in range(N)])
        labels = np.array([dist[i,:].argmin() for i in range(N)])

        centroids = np.array([[np.sum(A[labels==i, j])/np.sum(labels==i)
                                 for j in range(D)] for i in range(numCenter)])

    return centroids

def countParfors(test_func, args, **kws):
    typingctx = typing.Context()
    targetctx = cpu.CPUContext(typingctx)
    test_ir = compiler.run_frontend(test_func)
    if kws:
        options = cpu.ParallelOptions(kws)
    else:
        options = cpu.ParallelOptions(True)

    with cpu_target.nested_context(typingctx, targetctx):
        tp = TestPipeline(typingctx, targetctx, args, test_ir)

        inline_pass = inline_closurecall.InlineClosureCallPass(tp.func_ir, options)
        inline_pass.run()

        numba.rewrites.rewrite_registry.apply(
            'before-inference', tp, tp.func_ir)

        tp.typemap, tp.return_type, tp.calltypes = compiler.type_inference_stage(
            tp.typingctx, tp.func_ir, tp.args, None)

        type_annotations.TypeAnnotation(
            func_ir=tp.func_ir,
            typemap=tp.typemap,
            calltypes=tp.calltypes,
            lifted=(),
            lifted_from=None,
            args=tp.args,
            return_type=tp.return_type,
            html_output=config.HTML)

        preparfor_pass = numba.parfor.PreParforPass(
            tp.func_ir, tp.typemap, tp.calltypes, tp.typingctx, options)
        preparfor_pass.run()

        numba.rewrites.rewrite_registry.apply(
            'after-inference', tp, tp.func_ir)

        parfor_pass = numba.parfor.ParforPass(
            tp.func_ir, tp.typemap, tp.calltypes, tp.return_type,
            tp.typingctx, options)
        parfor_pass.run()
    ret_count = 0

    for label, block in test_ir.blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, numba.parfor.Parfor):
                ret_count += 1

    return ret_count


class TestPipeline(object):
    def __init__(self, typingctx, targetctx, args, test_ir):
        typingctx.refresh()
        targetctx.refresh()
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.args = args
        self.func_ir = test_ir
        self.typemap = None
        self.return_type = None
        self.calltypes = None


class TestParfors(TestParforsBase):

    def __init__(self, *args):
        TestParforsBase.__init__(self, *args)
        # these are used in the mass of simple tests
        m = np.reshape(np.arange(12.), (3, 4))
        self.simple_args = [np.arange(3.), np.arange(4.), m, m.T]

    def check(self, pyfunc, *args, **kwargs):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        self.check_prange_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    @skip_unsupported
    @tag('important')
    def test_arraymap(self):
        def test_impl(a, x, y):
            return a * x + y

        A = np.linspace(0, 1, 10)
        X = np.linspace(2, 1, 10)
        Y = np.linspace(1, 2, 10)

        self.check(test_impl, A, X, Y)

    @skip_unsupported
    @needs_blas
    @tag('important')
    def test_mvdot(self):
        def test_impl(a, v):
            return np.dot(a, v)

        A = np.linspace(0, 1, 20).reshape(2, 10)
        v = np.linspace(2, 1, 10)

        self.check(test_impl, A, v)

    @skip_unsupported
    @tag('important')
    def test_0d_broadcast(self):
        def test_impl():
            X = np.array(1)
            Y = np.ones((10, 12))
            return np.sum(X + Y)
        self.check(test_impl)
        self.assertTrue(countParfors(test_impl, ()) == 1)

    @skip_unsupported
    @tag('important')
    def test_2d_parfor(self):
        def test_impl():
            X = np.ones((10, 12))
            Y = np.zeros((10, 12))
            return np.sum(X + Y)
        self.check(test_impl)
        self.assertTrue(countParfors(test_impl, ()) == 1)

    @skip_unsupported
    @tag('important')
    def test_pi(self):
        def test_impl(n):
            x = 2 * np.random.ranf(n) - 1
            y = 2 * np.random.ranf(n) - 1
            return 4 * np.sum(x**2 + y**2 < 1) / n

        self.check(test_impl, 100000, decimal=1)
        self.assertTrue(countParfors(test_impl, (types.int64, )) == 1)

    @skip_unsupported
    @tag('important')
    def test_fuse_argmin(self):
        def test_impl(n):
            A = np.ones(n)
            C = A.argmin()
            B = A.sum()
            return B+C

        self.check(test_impl, 256)
        self.assertTrue(countParfors(test_impl, (types.int64, )) == 1)

    @skip_unsupported
    @tag('important')
    def test_test1(self):
        # blackscholes takes 5 1D float array args
        args = (numba.float64[:], ) * 5
        self.assertTrue(countParfors(test1, args) == 1)

    @skip_unsupported
    @needs_blas
    @tag('important')
    def test_test2(self):
        args = (numba.float64[:], numba.float64[:,:], numba.float64[:],
                numba.int64)
        self.assertTrue(countParfors(test2, args) == 1)

    @skip_unsupported
    @tag('important')
    def test_kmeans(self):
        np.random.seed(0)
        N = 1024
        D = 10
        centers = 3
        A = np.random.ranf((N, D))
        init_centroids = np.random.ranf((centers, D))
        self.check(test_kmeans_example, A, centers, 3, init_centroids,
                                                                    decimal=1)
        # TODO: count parfors after k-means fusion is working
        # requires recursive parfor counting

    @unittest.skipIf(not (_windows_py27 or _32bit),
                     "Only impacts Windows with Python 2.7 / 32 bit hardware")
    @needs_blas
    def test_unsupported_combination_raises(self):
        """
        This test is in place until issues with the 'parallel'
        target on Windows with Python 2.7 / 32 bit hardware are fixed.
        """

        with self.assertRaises(RuntimeError) as raised:
            @njit(parallel=True)
            def ddot(a, v):
                return np.dot(a, v)

            A = np.linspace(0, 1, 20).reshape(2, 10)
            v = np.linspace(2, 1, 10)
            ddot(A, v)

        msg = ("The 'parallel' target is not currently supported on "
               "Windows operating systems when using Python 2.7, "
               "or on 32 bit hardware")
        self.assertIn(msg, str(raised.exception))

    @skip_unsupported
    def test_simple01(self):
        def test_impl():
            return np.ones(())
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl)
        self.assertIn("\'@do_scheduling\' not found", str(raises.exception))

    @skip_unsupported
    def test_simple02(self):
        def test_impl():
            return np.ones((1,))
        self.check(test_impl)

    @skip_unsupported
    def test_simple03(self):
        def test_impl():
            return np.ones((1, 2))
        self.check(test_impl)

    @skip_unsupported
    def test_simple04(self):
        def test_impl():
            return np.ones(1)
        self.check(test_impl)

    @skip_unsupported
    def test_simple07(self):
        def test_impl():
            return np.ones((1, 2), dtype=np.complex128)
        self.check(test_impl)

    @skip_unsupported
    def test_simple08(self):
        def test_impl():
            return np.ones((1, 2)) + np.ones((1, 2))
        self.check(test_impl)

    @skip_unsupported
    def test_simple09(self):
        def test_impl():
            return np.ones((1, 1))
        self.check(test_impl)

    @skip_unsupported
    def test_simple10(self):
        def test_impl():
            return np.ones((0, 0))
        self.check(test_impl)

    @skip_unsupported
    def test_simple11(self):
        def test_impl():
            return np.ones((10, 10)) + 1.
        self.check(test_impl)

    @skip_unsupported
    def test_simple12(self):
        def test_impl():
            return np.ones((10, 10)) + np.complex128(1.)
        self.check(test_impl)

    @skip_unsupported
    def test_simple13(self):
        def test_impl():
            return np.complex128(1.)
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl)
        self.assertIn("\'@do_scheduling\' not found", str(raises.exception))

    @skip_unsupported
    def test_simple14(self):
        def test_impl():
            return np.ones((10, 10))[0::20]
        self.check(test_impl)

    @skip_unsupported
    def test_simple15(self):
        def test_impl(v1, v2, m1, m2):
            return v1 + v1
        self.check(test_impl, *self.simple_args)

    @skip_unsupported
    def test_simple16(self):
        def test_impl(v1, v2, m1, m2):
            return m1 + m1
        self.check(test_impl, *self.simple_args)

    @skip_unsupported
    def test_simple17(self):
        def test_impl(v1, v2, m1, m2):
            return m2 + v1
        self.check(test_impl, *self.simple_args)

    @skip_unsupported
    @needs_lapack
    def test_simple18(self):
        def test_impl(v1, v2, m1, m2):
            return m1.T + np.linalg.svd(m2)[1]
        self.check(test_impl, *self.simple_args)

    @skip_unsupported
    @needs_blas
    def test_simple19(self):
        def test_impl(v1, v2, m1, m2):
            return np.dot(m1, v2)
        self.check(test_impl, *self.simple_args)

    @skip_unsupported
    @needs_blas
    def test_simple20(self):
        def test_impl(v1, v2, m1, m2):
            return np.dot(m1, m2)
        # gemm is left to BLAS
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl, *self.simple_args)
        self.assertIn("\'@do_scheduling\' not found", str(raises.exception))

    @skip_unsupported
    @needs_blas
    def test_simple21(self):
        def test_impl(v1, v2, m1, m2):
            return np.dot(v1, v1)
        self.check(test_impl, *self.simple_args)

    @skip_unsupported
    def test_simple22(self):
        def test_impl(v1, v2, m1, m2):
            return np.sum(v1 + v1)
        self.check(test_impl, *self.simple_args)

    @skip_unsupported
    def test_simple23(self):
        def test_impl(v1, v2, m1, m2):
            x = 2 * v1
            y = 2 * v1
            return 4 * np.sum(x**2 + y**2 < 1) / 10
        self.check(test_impl, *self.simple_args)

    @skip_unsupported
    def test_np_func_direct_import(self):
        from numpy import ones  # import here becomes FreeVar
        def test_impl(n):
            A = ones(n)
            return A[0]
        n = 111
        self.check(test_impl, n)

    @skip_unsupported
    def test_np_random_func_direct_import(self):
        def test_impl(n):
            A = randn(n)
            return A[0]
        self.assertTrue(countParfors(test_impl, (types.int64, )) == 1)

    @skip_unsupported
    def test_arange(self):
        # test with stop only
        def test_impl1(n):
            return np.arange(n)
        # start and stop
        def test_impl2(s, n):
            return np.arange(n)
        # start, step, stop
        def test_impl3(s, n, t):
            return np.arange(s, n, t)

        for arg in [11, 128, 30.0, complex(4,5), complex(5,4)]:
            self.check(test_impl1, arg)
            self.check(test_impl2, 2, arg)
            self.check(test_impl3, 2, arg, 2)

    @skip_unsupported
    def test_linspace(self):
        # without num
        def test_impl1(start, stop):
            return np.linspace(start, stop)
        # with num
        def test_impl2(start, stop, num):
            return np.linspace(start, stop, num)

        for arg in [11, 128, 30.0, complex(4,5), complex(5,4)]:
            self.check(test_impl1, 2, arg)
            self.check(test_impl2, 2, arg, 30)

    @skip_unsupported
    def test_size_assertion(self):
        def test_impl(m, n):
            A = np.ones(m)
            B = np.ones(n)
            return np.sum(A + B)

        self.check(test_impl, 10, 10)
        with self.assertRaises(AssertionError) as raises:
            cfunc = njit(parallel=True)(test_impl)
            cfunc(10, 9)
        msg = "Sizes of A, B do not match"
        self.assertIn(msg, str(raises.exception))

    @skip_unsupported
    def test_mean(self):
        def test_impl(A):
            return A.mean()
        N = 100
        A = np.random.ranf(N)
        B = np.random.randint(10, size=(N, 3))
        self.check(test_impl, A)
        self.check(test_impl, B)
        self.assertTrue(countParfors(test_impl, (types.Array(types.float64, 1, 'C'), )) == 1)
        self.assertTrue(countParfors(test_impl, (types.Array(types.float64, 2, 'C'), )) == 1)

    @skip_unsupported
    def test_var(self):
        def test_impl(A):
            return A.var()
        N = 100
        A = np.random.ranf(N)
        B = np.random.randint(10, size=(N, 3))
        self.check(test_impl, A)
        self.check(test_impl, B)
        self.assertTrue(countParfors(test_impl, (types.Array(types.float64, 1, 'C'), )) == 2)
        self.assertTrue(countParfors(test_impl, (types.Array(types.float64, 2, 'C'), )) == 2)

    @skip_unsupported
    def test_std(self):
        def test_impl(A):
            return A.std()
        N = 100
        A = np.random.ranf(N)
        B = np.random.randint(10, size=(N, 3))
        self.check(test_impl, A)
        self.check(test_impl, B)
        self.assertTrue(countParfors(test_impl, (types.Array(types.float64, 1, 'C'), )) == 2)
        self.assertTrue(countParfors(test_impl, (types.Array(types.float64, 2, 'C'), )) == 2)

    @skip_unsupported
    def test_random_parfor(self):
        """
        Test function with only a random call to make sure a random function
        like ranf is actually translated to a parfor.
        """
        def test_impl(n):
            A = np.random.ranf((n, n))
            return A
        self.assertTrue(countParfors(test_impl, (types.int64, )) == 1)

    @skip_unsupported
    def test_randoms(self):
        def test_impl(n):
            A = np.random.standard_normal(size=(n, n))
            B = np.random.randn(n, n)
            C = np.random.normal(0.0, 1.0, (n, n))
            D = np.random.chisquare(1.0, (n, n))
            E = np.random.randint(1, high=3, size=(n, n))
            F = np.random.triangular(1, 2, 3, (n, n))
            return np.sum(A+B+C+D+E+F)

        n = 128
        cpfunc = self.compile_parallel(test_impl, (numba.typeof(n),))
        parfor_output = cpfunc.entry_point(n)
        py_output = test_impl(n)
        # check results within 5% since random numbers generated in parallel
        np.testing.assert_allclose(parfor_output, py_output, rtol=0.05)
        self.assertTrue(countParfors(test_impl, (types.int64, )) == 1)

    @skip_unsupported
    def test_dead_randoms(self):
        def test_impl(n):
            A = np.random.standard_normal(size=(n, n))
            B = np.random.randn(n, n)
            C = np.random.normal(0.0, 1.0, (n, n))
            D = np.random.chisquare(1.0, (n, n))
            E = np.random.randint(1, high=3, size=(n, n))
            F = np.random.triangular(1, 2, 3, (n, n))
            return 3

        n = 128
        cpfunc = self.compile_parallel(test_impl, (numba.typeof(n),))
        parfor_output = cpfunc.entry_point(n)
        py_output = test_impl(n)
        self.assertEqual(parfor_output, py_output)
        self.assertTrue(countParfors(test_impl, (types.int64, )) == 0)

    @skip_unsupported
    def test_cfg(self):
        # from issue #2477
        def test_impl(x, is_positive, N):
            for i in numba.prange(2):
                for j in range( i*N//2, (i+1)*N//2 ):
                    is_positive[j] = 0
                    if x[j] > 0:
                        is_positive[j] = 1

            return is_positive

        N = 100
        x = np.random.rand(N)
        is_positive = np.zeros(N)
        self.check(test_impl, x, is_positive, N)

    @skip_unsupported
    def test_reduce(self):
        def test_impl(A):
            init_val = 10
            return reduce(lambda a,b: min(a, b), A, init_val)

        n = 211
        A = np.random.ranf(n)
        self.check(test_impl, A)
        A = np.random.randint(10, size=n).astype(np.int32)
        self.check(test_impl, A)

        # test checking the number of arguments for the reduce function
        def test_impl():
            g = lambda x: x ** 2
            return reduce(g, np.array([1, 2, 3, 4, 5]), 2)
        with self.assertTypingError():
            self.check(test_impl)

        # test checking reduction over bitarray masked arrays
        n = 160
        A = np.random.randint(10, size=n).astype(np.int32)
        def test_impl(A):
            return np.sum(A[A>=3])
        self.check(test_impl, A)
        # TODO: this should fuse
        # self.assertTrue(countParfors(test_impl, (numba.float64[:],)) == 1)

        def test_impl(A):
            B = A[:,0]
            return np.sum(A[B>=3,1])
        self.check(test_impl, A.reshape((16,10)))
        # TODO: this should also fuse
        #self.assertTrue(countParfors(test_impl, (numba.float64[:,:],)) == 1)

        def test_impl(A):
            B = A[:,0]
            return np.sum(A[B>=3,1:2])
        self.check(test_impl, A.reshape((16,10)))
        # this doesn't fuse due to mixed indices
        self.assertTrue(countParfors(test_impl, (numba.float64[:,:],)) == 2)

    @skip_unsupported
    def test_min(self):
        def test_impl1(A):
            return A.min()

        def test_impl2(A):
            return np.min(A)

        n = 211
        A = np.random.ranf(n)
        B = np.random.randint(10, size=n).astype(np.int32)
        C = np.random.ranf((n, n))  # test multi-dimensional array
        self.check(test_impl1, A)
        self.check(test_impl1, B)
        self.check(test_impl1, C)
        self.check(test_impl2, A)
        self.check(test_impl2, B)
        self.check(test_impl2, C)

    @skip_unsupported
    def test_max(self):
        def test_impl1(A):
            return A.max()

        def test_impl2(A):
            return np.max(A)

        n = 211
        A = np.random.ranf(n)
        B = np.random.randint(10, size=n).astype(np.int32)
        C = np.random.ranf((n, n))  # test multi-dimensional array
        self.check(test_impl1, A)
        self.check(test_impl1, B)
        self.check(test_impl1, C)
        self.check(test_impl2, A)
        self.check(test_impl2, B)
        self.check(test_impl2, C)

    @skip_unsupported
    def test_argmin(self):
        def test_impl1(A):
            return A.argmin()

        def test_impl2(A):
            return np.argmin(A)

        n = 211
        A = np.array([1., 0., 2., 0., 3.])
        B = np.random.randint(10, size=n).astype(np.int32)
        C = np.random.ranf((n, n))  # test multi-dimensional array
        self.check(test_impl1, A)
        self.check(test_impl1, B)
        self.check(test_impl1, C)
        self.check(test_impl2, A)
        self.check(test_impl2, B)
        self.check(test_impl2, C)

    @skip_unsupported
    def test_argmax(self):
        def test_impl1(A):
            return A.argmax()

        def test_impl2(A):
            return np.argmax(A)

        n = 211
        A = np.array([1., 0., 3., 2., 3.])
        B = np.random.randint(10, size=n).astype(np.int32)
        C = np.random.ranf((n, n))  # test multi-dimensional array
        self.check(test_impl1, A)
        self.check(test_impl1, B)
        self.check(test_impl1, C)
        self.check(test_impl2, A)
        self.check(test_impl2, B)
        self.check(test_impl2, C)

class TestPrange(TestParforsBase):

    def prange_tester(self, pyfunc, *args, **kwargs):
        """
        The `prange` tester
        This is a hack. It basically switches out range calls for prange.
        It does this by copying the live code object of a function
        containing 'range' then copying the .co_names and mutating it so
        that 'range' is replaced with 'prange'. It then creates a new code
        object containing the mutation and instantiates a function to contain
        it. At this point three results are created:
        1. The result of calling the original python function.
        2. The result of calling a njit compiled version of the original
            python function.
        3. The result of calling a njit(parallel=True) version of the mutated
           function containing `prange`.
        The three results are then compared and the `prange` based function's
        llvm_ir is inspected to ensure the scheduler code is present.

        Arguments:
         pyfunc - the python function to test
         args - data arguments to pass to the pyfunc under test

        Keyword Arguments:
         patch_instance - iterable containing which instances of `range` to
                          replace. If not present all instance of `range` are
                          replaced.
         Remaining kwargs are passed to np.testing.assert_almost_equal


        Example:
            def foo():
                acc = 0
                for x in range(5):
                    for y in range(10):
                        acc +=1
                return acc

            # calling as
            prange_tester(foo)
            # will test code equivalent to
            # def foo():
            #     acc = 0
            #     for x in prange(5): # <- changed
            #         for y in prange(10): # <- changed
            #             acc +=1
            #     return acc

            # calling as
            prange_tester(foo, patch_instance=[1])
            # will test code equivalent to
            # def foo():
            #     acc = 0
            #     for x in range(5): # <- outer loop (0) unchanged
            #         for y in prange(10): # <- inner loop (1) changed
            #             acc +=1
            #     return acc

        """

        pyfunc_code = pyfunc.__code__

        prange_names = list(pyfunc_code.co_names)

        patch_instance = kwargs.pop('patch_instance', None)
        if not patch_instance:
            # patch all instances, cheat by just switching
            # range for prange
            assert 'range' in pyfunc_code.co_names
            prange_names = tuple([x if x != 'range' else 'prange'
                                  for x in pyfunc_code.co_names])
            new_code = bytes(pyfunc_code.co_code)
        else:
            # patch specified instances...
            # find where 'range' is in co_names
            range_idx = pyfunc_code.co_names.index('range')
            range_locations = []
            # look for LOAD_GLOBALs that point to 'range'
            for _, instr in ByteCodeIter(pyfunc_code):
                if instr.opname == 'LOAD_GLOBAL':
                    if instr.arg == range_idx:
                        range_locations.append(instr.offset + 1)
            # add in 'prange' ref
            prange_names.append('prange')
            prange_names = tuple(prange_names)
            prange_idx = len(prange_names) - 1
            new_code = bytearray(pyfunc_code.co_code)
            assert len(patch_instance) <= len(range_locations)
            # patch up the new byte code
            for i in patch_instance:
                idx = range_locations[i]
                new_code[idx] = prange_idx
            new_code = bytes(new_code)

        # create new code parts
        co_args = [pyfunc_code.co_argcount]
        if sys.version_info > (3, 0):
            co_args.append(pyfunc_code.co_kwonlyargcount)
        co_args.extend([pyfunc_code.co_nlocals,
                        pyfunc_code.co_stacksize,
                        pyfunc_code.co_flags,
                        new_code,
                        pyfunc_code.co_consts,
                        prange_names,
                        pyfunc_code.co_varnames,
                        pyfunc_code.co_filename,
                        pyfunc_code.co_name,
                        pyfunc_code.co_firstlineno,
                        pyfunc_code.co_lnotab,
                        pyfunc_code.co_freevars,
                        pyfunc_code.co_cellvars
                        ])

        # create code object with prange mutation
        prange_code = pytypes.CodeType(*co_args)

        # get function
        pfunc = pytypes.FunctionType(prange_code, globals())

        # Compile functions
        # compile a standard njit of the original function
        sig = tuple([numba.typeof(x) for x in args])
        cfunc = self.compile_njit(pyfunc, sig)

        # compile the prange injected function
        cpfunc = self.compile_parallel(pfunc, sig)

        # compare
        self.check_prange_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    @skip_unsupported
    def test_prange01(self):
        def test_impl():
            n = 4
            A = np.zeros(n)
            for i in range(n):
                A[i] = 2.0 * i
            return A
        self.prange_tester(test_impl)

    @skip_unsupported
    def test_prange02(self):
        def test_impl():
            n = 4
            A = np.zeros(n - 1)
            for i in range(1, n):
                A[i - 1] = 2.0 * i
            return A
        self.prange_tester(test_impl)

    @skip_unsupported
    def test_prange03(self):
        def test_impl():
            s = 0
            for i in range(10):
                s += 2
            return s
        self.prange_tester(test_impl)

    @skip_unsupported
    def test_prange04(self):
        def test_impl():
            a = 2
            b = 3
            A = np.empty(4)
            for i in range(4):
                if i == a:
                    A[i] = b
                else:
                    A[i] = 0
            return A
        self.prange_tester(test_impl)

    @skip_unsupported
    def test_prange05(self):
        def test_impl():
            n = 4
            A = np.ones((n), dtype=np.float64)
            s = 0
            for i in range(1, n - 1, 1):
                s += A[i]
            return s
        self.prange_tester(test_impl)

    @skip_unsupported
    def test_prange06(self):
        def test_impl():
            n = 4
            A = np.ones((n), dtype=np.float64)
            s = 0
            for i in range(1, 1, 1):
                s += A[i]
            return s
        self.prange_tester(test_impl)

    @skip_unsupported
    def test_prange07(self):
        def test_impl():
            n = 4
            A = np.ones((n), dtype=np.float64)
            s = 0
            for i in range(n, 1):
                s += A[i]
            return s
        self.prange_tester(test_impl)

    @skip_unsupported
    def test_prange08(self):
        def test_impl():
            n = 4
            A = np.ones((n))
            acc = 0
            for i in range(len(A)):
                for j in range(len(A)):
                    acc += A[i]
            return acc

        test_impl()

    @skip_unsupported
    def test_prange08_1(self):
        def test_impl():
            n = 4
            A = np.ones((n))
            acc = 0
            for i in range(4):
                for j in range(4):
                    acc += A[i]
            return acc
        self.prange_tester(test_impl)

    @skip_unsupported
    def test_prange09(self):
        def test_impl():
            n = 4
            acc = 0
            for i in range(n):
                for j in range(n):
                    acc += 1
            return acc
        # patch inner loop to 'prange'
        self.prange_tester(test_impl, patch_instance=[1])

    @skip_unsupported
    def test_prange10(self):
        def test_impl():
            n = 4
            acc2 = 0
            for j in range(n):
                acc1 = 0
                for i in range(n):
                    acc1 += 1
                acc2 += acc1
            return acc2
        # patch outer loop to 'prange'
        self.prange_tester(test_impl, patch_instance=[0])

    @skip_unsupported
    @unittest.skip("list append is not thread-safe yet (#2391, #2408)")
    def test_prange11(self):
        def test_impl():
            n = 4
            return [np.sin(j) for j in range(n)]
        self.prange_tester(test_impl)

    @skip_unsupported
    def test_prange12(self):
        def test_impl():
            acc = 0
            n = 4
            X = np.ones(n)
            for i in range(-len(X)):
                acc += X[i]
            return acc
        self.prange_tester(test_impl)

    @skip_unsupported
    def test_prange13(self):
        def test_impl(n):
            acc = 0
            for i in range(n):
                acc += 1
            return acc
        self.prange_tester(test_impl, np.int32(4))

    @skip_unsupported
    def test_prange14(self):
        def test_impl(A):
            s = 3
            for i in range(len(A)):
                s += A[i]*2
            return s
        # this tests reduction detection well since the accumulated variable
        # is initialized before the parfor and the value accessed from the array
        # is updated before accumulation
        self.prange_tester(test_impl, np.random.ranf(4))

    @skip_unsupported
    def test_prange15(self):
        # from issue 2587
        # test parfor type inference when there is multi-dimensional indexing
        def test_impl(N):
            acc = 0
            for i in range(N):
                x = np.ones((1, 1))
                acc += x[0, 0]
            return acc

        self.prange_tester(test_impl, 1024)

    @skip_unsupported
    def test_kde_example(self):
        def test_impl(X):
            # KDE example
            b = 0.5
            points = np.array([-1.0, 2.0, 5.0])
            N = points.shape[0]
            n = X.shape[0]
            exps = 0
            for i in range(n):
                p = X[i]
                d = (-(p - points)**2) / (2 * b**2)
                m = np.min(d)
                exps += m - np.log(b * N) + np.log(np.sum(np.exp(d - m)))
            return exps

        n = 128
        X = np.random.ranf(n)
        self.prange_tester(test_impl, X)

    @skip_unsupported
    def test_parfor_alias1(self):
        def test_impl(n):
            b = np.zeros((n, n))
            a = b[0]
            for j in range(n):
                a[j] = j + 1
            return b.sum()
        self.prange_tester(test_impl, 4)

    @skip_unsupported
    def test_parfor_alias2(self):
        def test_impl(n):
            b = np.zeros((n, n))
            for i in range(n):
              a = b[i]
              for j in range(n):
                a[j] = i + j
            return b.sum()
        self.prange_tester(test_impl, 4)

    @skip_unsupported
    def test_parfor_alias3(self):
        def test_impl(n):
            b = np.zeros((n, n, n))
            for i in range(n):
              a = b[i]
              for j in range(n):
                c = a[j]
                for k in range(n):
                  c[k] = i + j + k
            return b.sum()
        self.prange_tester(test_impl, 4)

class TestParforsSlice(TestParforsBase):

    def check(self, pyfunc, *args, **kwargs):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        self.check_prange_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    @skip_unsupported
    def test_parfor_slice1(self):
        def test_impl(a):
            (n,) = a.shape
            b = a[0:n-2] + a[1:n-1]
            return b

        self.check(test_impl, np.ones(10))

    @skip_unsupported
    def test_parfor_slice2(self):
        def test_impl(a, m):
            (n,) = a.shape
            b = a[0:n-2] + a[1:m]
            return b

        # runtime assertion should succeed
        self.check(test_impl, np.ones(10), 9)
        # next we expect failure
        with self.assertRaises(AssertionError) as raises:
            njit(parallel=True)(test_impl)(np.ones(10),10)
        self.assertIn("do not match", str(raises.exception))

    @skip_unsupported
    def test_parfor_slice3(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a[0:m-1,0:n-1] + a[1:m,1:n]
            return b

        self.check(test_impl, np.ones((4,3)))

    @skip_unsupported
    def test_parfor_slice4(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a[:,0:n-1] + a[:,1:n]
            return b

        self.check(test_impl, np.ones((4,3)))

    @skip_unsupported
    def test_parfor_slice5(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a[0:m-1,:] + a[1:m,:]
            return b

        self.check(test_impl, np.ones((4,3)))

    @skip_unsupported
    def test_parfor_slice6(self):
        def test_impl(a):
            b = a.transpose()
            c = a[1,:] + b[:,1]
            return c

        self.check(test_impl, np.ones((4,3)))

    @skip_unsupported
    def test_parfor_slice7(self):
        def test_impl(a):
            b = a.transpose()
            c = a[1,:] + b[1,:]
            return c

        # runtime check should succeed
        self.check(test_impl, np.ones((3,3)))
        # next we expect failure
        with self.assertRaises(AssertionError) as raises:
            njit(parallel=True)(test_impl)(np.ones((3,4)))
        self.assertIn("do not match", str(raises.exception))

    @skip_unsupported
    def test_parfor_slice8(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.transpose()
            b[1:m,1:n] = a[1:m,1:n]
            return b

        self.check(test_impl, np.arange(9).reshape((3,3)))

    @skip_unsupported
    def test_parfor_slice9(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.transpose()
            b[1:n,1:m] = a[:,1:m]
            return b

        self.check(test_impl, np.arange(12).reshape((3,4)))

    @skip_unsupported
    def test_parfor_slice10(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.transpose()
            b[2,1:m] = a[2,1:m]
            return b

        self.check(test_impl, np.arange(9).reshape((3,3)))

    @skip_unsupported
    def test_parfor_slice11(self):
        def test_impl(a):
            (m,n,l) = a.shape
            b = a.copy()
            b[:,1,1:l] = a[:,2,1:l]
            return b

        self.check(test_impl, np.arange(27).reshape((3,3,3)))

    @skip_unsupported
    def test_parfor_slice12(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.copy()
            b[1,1:-1] = a[0,:-2]
            return b

        self.check(test_impl, np.arange(12).reshape((3,4)))

    @skip_unsupported
    def test_parfor_slice13(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.copy()
            c = -1
            b[1,1:c] = a[0,-n:c-1]
            return b

        self.check(test_impl, np.arange(12).reshape((3,4)))

    @skip_unsupported
    def test_parfor_slice14(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.copy()
            c = -1
            b[1,:-1] = a[0,-3:4]
            return b

        self.check(test_impl, np.arange(12).reshape((3,4)))

    @skip_unsupported
    def test_parfor_slice15(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.copy()
            c = -1
            b[1,-(n-1):] = a[0,-3:4]
            return b

        self.check(test_impl, np.arange(12).reshape((3,4)))

    @skip_unsupported
    def test_parfor_slice16(self):
        def test_impl(a, b, n):
            assert(a.shape == b.shape)
            a[1:n] = 10
            b[0:(n-1)] = 10
            return a * b

        self.check(test_impl, np.ones(10), np.zeros(10), 8)
        args = (numba.float64[:], numba.float64[:], numba.int64)
        self.assertEqual(countParfors(test_impl, args), 2)

    @skip_unsupported
    def test_parfor_slice17(self):
        def test_impl(m, A):
            B = np.zeros(m)
            n = len(A)
            B[-n:] = A
            return B

        self.check(test_impl, 10, np.ones(10))


class TestParforsOptions(TestParforsBase):

    def check(self, pyfunc, *args, **kwargs):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        self.check_prange_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    @skip_unsupported
    def test_parfor_options(self):
        def test_impl(a):
            n = a.shape[0]
            b = np.ones(n)
            c = np.array([ i for i in range(n) ])
            b[:n] = a + b * c
            for i in prange(n):
                c[i] = b[i] * a[i]
            return reduce(lambda x,y:x+y, c, 0)

        self.check(test_impl, np.ones(10))
        args = (numba.float64[:],)
        # everything should fuse with default option
        self.assertEqual(countParfors(test_impl, args), 1)
        # with no fusion
        self.assertEqual(countParfors(test_impl, args, fusion=False), 6)
        # with no fusion, comprehension
        self.assertEqual(countParfors(test_impl, args, fusion=False,
                         comprehension=False), 5)
        #with no fusion, comprehension, setitem
        self.assertEqual(countParfors(test_impl, args, fusion=False,
                         comprehension=False, setitem=False), 4)
         # with no fusion, comprehension, prange
        self.assertEqual(countParfors(test_impl, args, fusion=False,
                         comprehension=False, setitem=False, prange=False), 3)
         # with no fusion, comprehension, prange, reduction
        self.assertEqual(countParfors(test_impl, args, fusion=False,
                         comprehension=False, setitem=False, prange=False,
                         reduction=False), 2)
        # with no fusion, comprehension, prange, reduction, numpy
        self.assertEqual(countParfors(test_impl, args, fusion=False,
                         comprehension=False, setitem=False, prange=False,
                         reduction=False, numpy=False), 0)


class TestParforsBitMask(TestParforsBase):

    def check(self, pyfunc, *args, **kwargs):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        self.check_prange_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    @skip_unsupported
    def test_parfor_bitmask1(self):
        def test_impl(a, n):
            b = a > n
            a[b] = 0
            return a

        self.check(test_impl, np.arange(10), 5)

    @skip_unsupported
    def test_parfor_bitmask2(self):
        def test_impl(a, b):
            a[b] = 0
            return a

        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    @skip_unsupported
    def test_parfor_bitmask3(self):
        def test_impl(a, b):
            a[b] = a[b]
            return a

        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    @skip_unsupported
    def test_parfor_bitmask4(self):
        def test_impl(a, b):
            a[b] = (2 * a)[b]
            return a

        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    @skip_unsupported
    def test_parfor_bitmask5(self):
        def test_impl(a, b):
            a[b] = a[b] * a[b]
            return a

        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    @skip_unsupported
    def test_parfor_bitmask6(self):
        def test_impl(a, b, c):
            a[b] = c
            return a

        a = np.arange(10)
        b = a > 5
        c = np.zeros(sum(b))

        # expect failure due to lack of parallelism
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl, a, b, c)
        self.assertIn("\'@do_scheduling\' not found", str(raises.exception))

class TestParforsMisc(TestCase):
    """
    Tests miscellaneous parts of ParallelAccelerator use.
    """

    @skip_unsupported
    def test_warn_if_cache_set(self):

        def pyfunc():
            return

        with warnings.catch_warnings(record=True) as raised_warnings:
            warnings.simplefilter('always')
            cfunc = njit(parallel=True, cache=True)(pyfunc)
            cfunc()

        self.assertEqual(len(raised_warnings), 1)

        warning_obj = raised_warnings[0]

        expected_msg = ("Caching is not available when the 'parallel' target "
                        "is in use. Caching is now being disabled to allow "
                        "execution to continue.")

        # check warning message appeared
        self.assertIn(expected_msg, str(warning_obj.message))

        # make sure the cache is set to false, cf. NullCache
        self.assertTrue(isinstance(cfunc._cache, numba.caching.NullCache))

if __name__ == "__main__":
    unittest.main()
