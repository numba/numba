#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

from __future__ import print_function, division, absolute_import

import math
import re
import sys
import types as pytypes
import warnings

import numpy as np

import numba
from numba import unittest_support as unittest
from numba import njit, prange
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


class TestParforsBase(unittest.TestCase):
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
        self.pflags.set('auto_parallel')
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


def countParfors(func_ir):
    ret_count = 0

    for label, block in func_ir.blocks.items():
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
    def test_2d_parfor(self):
        def test_impl():
            X = np.ones((10, 12))
            Y = np.zeros((10, 12))
            return np.sum(X + Y)
        self.check(test_impl)

    @skip_unsupported
    @tag('important')
    def test_pi(self):
        def test_impl(n):
            x = 2 * np.random.ranf(n) - 1
            y = 2 * np.random.ranf(n) - 1
            return 4 * np.sum(x**2 + y**2 < 1) / n

        self.check(test_impl, 100000, decimal=1)

    @skip_unsupported
    @tag('important')
    def test_test1(self):
        typingctx = typing.Context()
        targetctx = cpu.CPUContext(typingctx)
        test_ir = compiler.run_frontend(test1)
        with cpu_target.nested_context(typingctx, targetctx):
            one_arg = numba.types.npytypes.Array(
                numba.types.scalars.Float(name="float64"), 1, 'C')
            args = (one_arg, one_arg, one_arg, one_arg, one_arg)
            tp = TestPipeline(typingctx, targetctx, args, test_ir)

            numba.rewrites.rewrite_registry.apply(
                'before-inference', tp, tp.func_ir)

            tp.typemap, tp.return_type, tp.calltypes = compiler.type_inference_stage(
                tp.typingctx, tp.func_ir, tp.args, None)

            type_annotation = type_annotations.TypeAnnotation(
                func_ir=tp.func_ir,
                typemap=tp.typemap,
                calltypes=tp.calltypes,
                lifted=(),
                lifted_from=None,
                args=tp.args,
                return_type=tp.return_type,
                html_output=config.HTML)

            numba.rewrites.rewrite_registry.apply(
                'after-inference', tp, tp.func_ir)

            parfor_pass = numba.parfor.ParforPass(
                tp.func_ir, tp.typemap, tp.calltypes, tp.return_type,
                tp.typingctx)
            parfor_pass.run()
            self.assertTrue(countParfors(test_ir) == 1)

    @skip_unsupported
    @needs_blas
    @tag('important')
    def test_test2(self):
        typingctx = typing.Context()
        targetctx = cpu.CPUContext(typingctx)
        test_ir = compiler.run_frontend(test2)
        with cpu_target.nested_context(typingctx, targetctx):
            oneD_arg = numba.types.npytypes.Array(
                numba.types.scalars.Float(name="float64"), 1, 'C')
            twoD_arg = numba.types.npytypes.Array(
                numba.types.scalars.Float(name="float64"), 2, 'C')
            args = (oneD_arg, twoD_arg, oneD_arg, types.int64)
            tp = TestPipeline(typingctx, targetctx, args, test_ir)

            numba.rewrites.rewrite_registry.apply(
                'before-inference', tp, tp.func_ir)

            tp.typemap, tp.return_type, tp.calltypes = compiler.type_inference_stage(
                tp.typingctx, tp.func_ir, tp.args, None)

            type_annotation = type_annotations.TypeAnnotation(
                func_ir=tp.func_ir,
                typemap=tp.typemap,
                calltypes=tp.calltypes,
                lifted=(),
                lifted_from=None,
                args=tp.args,
                return_type=tp.return_type,
                html_output=config.HTML)

            numba.rewrites.rewrite_registry.apply(
                'after-inference', tp, tp.func_ir)

            parfor_pass = numba.parfor.ParforPass(
                tp.func_ir, tp.typemap, tp.calltypes, tp.return_type,
                tp.typingctx)
            parfor_pass.run()
            self.assertTrue(countParfors(test_ir) == 1)

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
            return m1 + np.linalg.svd(m2)[0][:-1, :]
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
        self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())

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

class TestParforsMisc(unittest.TestCase):
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
