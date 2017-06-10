#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

from __future__ import print_function, division, absolute_import

import sys
import re

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
from numba.ir_utils import copy_propagate, apply_copy_propagate, get_name_var_table, remove_dels, remove_dead
from numba import ir

import math

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
    put  = call - futureValue + sptprice
    return put

def test2(Y,X,w,iterations):
    # logistic regression example
    for i in range(iterations):
        w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X,w))) - 1.0) * Y),X)
    return w

def kde_example(X):
    # KDE example
    b = 0.5
    points = np.array([-1.0, 2.0, 5.0])
    N = points.shape[0]
    n = X.shape[0]
    exps = 0
    for i in prange(n):
        p = X[i]
        d = (-(p-points)**2)/(2*b**2)
        m = np.min(d)
        exps += m-np.log(b*N)+np.log(np.sum(np.exp(d-m)))
    return exps

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

class TestParfors(unittest.TestCase):

    def test_arraymap(self):
        @njit(parallel=True)
        def axy(a, x, y):
            return a * x + y

        A = np.linspace(0,1,10)
        X = np.linspace(2,1,10)
        Y = np.linspace(1,2,10)

        output = axy(A,X,Y)
        expected = A*X+Y
        np.testing.assert_array_equal(expected, output)
        self.assertIn('@do_scheduling', axy.inspect_llvm(axy.signatures[0]))

    def test_mvdot(self):
        @njit(parallel=True)
        def ddot(a, v):
            return np.dot(a,v)

        A = np.linspace(0,1,20).reshape(2,10)
        v = np.linspace(2,1,10)

        output = ddot(A,v)
        expected = np.dot(A,v)
        np.testing.assert_array_almost_equal(expected, output, decimal=5)
        self.assertIn('@do_scheduling', ddot.inspect_llvm(ddot.signatures[0]))

    def test_2d_parfor(self):
        @njit(parallel=True)
        def test_2d():
            X = np.ones((10,12))
            Y = np.zeros((10,12))
            return np.sum(X+Y)

        output = test_2d()
        expected = 120.0
        np.testing.assert_almost_equal(expected, output)
        self.assertIn('@do_scheduling', test_2d.inspect_llvm(test_2d.signatures[0]))

    def test_prange1(self):
        @numba.njit(parallel=True)
        def test_p1(n):
            A = np.zeros(n)
            for i in prange(n):
                A[i] = 2.0*i
            return A

        output = test_p1(4)
        expected = np.array([ 0.,  2.,  4.,  6.])
        np.testing.assert_array_almost_equal(expected, output)
        self.assertIn('@do_scheduling', test_p1.inspect_llvm(test_p1.signatures[0]))

    def test_prange2(self):
        @numba.njit(parallel=True)
        def test_p2(n):
            A = np.zeros(n-1)
            for i in numba.prange(1,n):
                A[i-1] = 2.0*i
            return A

        output = test_p2(4)
        expected = np.array([2., 4.,  6.])
        np.testing.assert_array_almost_equal(expected, output)
        self.assertIn('@do_scheduling', test_p2.inspect_llvm(test_p2.signatures[0]))

    def test_prange3(self):
        @numba.njit(parallel=True)
        def test_p3():
            s = 0
            for i in prange(10):
                s += 2
            return s

        output = test_p3()
        expected = 20
        np.testing.assert_almost_equal(expected, output)
        self.assertIn('@do_scheduling', test_p3.inspect_llvm(test_p3.signatures[0]))

    def test_prange4(self):
        @numba.njit(parallel=True)
        def test_p4():
            a = 2
            b = 3
            A = np.empty(4)
            for i in numba.prange(4):
                if i==a:
                    A[i] = b
                else:
                    A[i] = 0
            return A

        output = test_p4()
        expected = np.array([0., 0., 3., 0.])
        np.testing.assert_array_almost_equal(expected, output)
        self.assertIn('@do_scheduling', test_p4.inspect_llvm(test_p4.signatures[0]))

    def test_prange5(self):
        @numba.njit(parallel=True)
        def test_p5(A):
            s = 0
            for i in prange(1, n - 1, 1):
                s += A[i]
            return s
        n = 4
        A = np.ones((n), dtype=np.float64)
        output = test_p5(A)
        expected = 2.0
        np.testing.assert_almost_equal(expected, output)
        self.assertIn('@do_scheduling', test_p5.inspect_llvm(test_p5.signatures[0]))

    def test_prange6(self):
        @numba.njit(parallel=True)
        def test_p6(A):
            s = 0
            for i in prange(1, 1, 1):
                s += A[i]
            return s
        n = 4
        A = np.ones((n), dtype=np.float64)
        output = test_p6(A)
        expected = 0.0
        np.testing.assert_almost_equal(expected, output)
        self.assertIn('@do_scheduling', test_p6.inspect_llvm(test_p6.signatures[0]))

    def test_prange7(self):
        @numba.njit(parallel=True)
        def test_p7(A):
            s = 0
            for i in prange(n, 1):
                s += A[i]
            return s
        n = 4
        A = np.ones((n), dtype=np.float64)
        output = test_p7(A)
        expected = 0.0
        np.testing.assert_almost_equal(expected, output)
        self.assertIn('@do_scheduling', test_p7.inspect_llvm(test_p7.signatures[0]))

    def test_prange8(self):
        @numba.njit(parallel=True)
        def test_p8(A):
            acc = 0
            for i in prange(len(A)):
                for j in prange(len(A)):
                    acc+=A[i]
            return acc
        n=4
        A = np.ones((n))
        output = test_p8(A)
        expected = 16.0
        np.testing.assert_almost_equal(output, expected)
        self.assertIn('@do_scheduling', test_p8.inspect_llvm(test_p8.signatures[0]))

    def test_prange8_1(self):
        @numba.njit(parallel=True)
        def test_p8(A):
            acc = 0
            for i in prange(4):
                for j in prange(4):
                    acc+=A[i]
            return acc
        n=4
        A = np.ones((n))
        output = test_p8(A)
        expected = 16.0
        np.testing.assert_almost_equal(output, expected)
        self.assertIn('@do_scheduling', test_p8.inspect_llvm(test_p8.signatures[0]))

    def test_prange9(self):
        # does this count as cross iteration dependency?
        # the inner parallel loop should reduce on acc
        # for each outer loop?
        @numba.njit(parallel=True)
        def test_p9(n):
            acc = 0
            for i in range(n):
                for j in prange(n):
                    acc+=1
            return acc
        n=4
        output = test_p9(n)
        expected = 16
        np.testing.assert_almost_equal(output, expected)


    def test_prange10(self):
        @numba.njit(parallel=True)
        def test_p10(NOT_USED):
            acc2 = 0
            for j in prange(n):
                acc1 = 0
                for i in range(n):
                    acc1 += 1
                acc2 += acc1
            return acc2

        n=4
        output = test_p10(n)
        expected = 16.
        np.testing.assert_almost_equal(output, expected)
        self.assertIn('@do_scheduling', test_p10.inspect_llvm(test_p10.signatures[0]))


    def test_prange11(self):
        ## List comprehension with a `prange` fails with
        ## `No definition for lowering <class 'numba.parfor.prange'>(int64,) -> range_state_int64`.
        @numba.njit(parallel=True)
        def test_p11(n):
            return [np.sin(j) for j in prange(n)]

        n=4
        output = test_p11(n)


    def test_prange12(self):
        # segfaults/hangs
        @numba.njit(parallel=True)
        def test_p12(X):
            acc = 0
            for i in prange(-len(X)):
                acc+=X[i]
            return acc

        n=4
        output = test_p12(np.ones(n))
        expected = 0
        np.testing.assert_almost_equal(output, expected)
        self.assertIn('@do_scheduling', test_p12.inspect_llvm(test_p12.signatures[0]))


    def test_prange13(self):
        # fails, Operands must be the same type, got (i32, i64)
        @numba.njit(parallel=True)
        def test_p13(n):
            acc = 0
            for i in prange(n):
                acc+=1
            return acc
        n=4
        output = test_p13(np.int32(n))


    def test_pi(self):
        @njit(parallel=True)
        def calc_pi(n):
            x = 2*np.random.ranf(n)-1
            y = 2*np.random.ranf(n)-1
            return 4*np.sum(x**2+y**2<1)/n

        output = calc_pi(100000)
        expected = 3.14
        np.testing.assert_almost_equal(expected, output, decimal=1)
        self.assertIn('@do_scheduling', calc_pi.inspect_llvm(calc_pi.signatures[0]))

    def test_test1(self):
        typingctx = typing.Context()
        targetctx = cpu.CPUContext(typingctx)
        test_ir = compiler.run_frontend(test1)
        #print("Num blocks = ", len(test_ir.blocks))
        #print(test_ir.dump())
        with cpu_target.nested_context(typingctx, targetctx):
            one_arg = numba.types.npytypes.Array(numba.types.scalars.Float(name="float64"), 1, 'C')
            args = (one_arg, one_arg, one_arg, one_arg, one_arg)
            #print("args = ", args)
            tp = TestPipeline(typingctx, targetctx, args, test_ir)

            numba.rewrites.rewrite_registry.apply('before-inference', tp, tp.func_ir)

            tp.typemap, tp.return_type, tp.calltypes = compiler.type_inference_stage(tp.typingctx, tp.func_ir, tp.args, None)
            #print("typemap = ", tp.typemap)
            #print("return_type = ", tp.return_type)

            type_annotation = type_annotations.TypeAnnotation(
                func_ir=tp.func_ir,
                typemap=tp.typemap,
                calltypes=tp.calltypes,
                lifted=(),
                lifted_from=None,
                args=tp.args,
                return_type=tp.return_type,
                html_output=config.HTML)

            numba.rewrites.rewrite_registry.apply('after-inference', tp, tp.func_ir)

            parfor_pass = numba.parfor.ParforPass(tp.func_ir, tp.typemap, tp.calltypes, tp.return_type)
            parfor_pass.run()
            #print(tp.func_ir.dump())
            #print(countParfors(tp.func_ir) == 1)
            self.assertTrue(countParfors(test_ir) == 1)

    def test_test2(self):
        typingctx = typing.Context()
        targetctx = cpu.CPUContext(typingctx)
        test_ir = compiler.run_frontend(test2)
        #print("Num blocks = ", len(test_ir.blocks))
        #print(test_ir.dump())
        with cpu_target.nested_context(typingctx, targetctx):
            oneD_arg = numba.types.npytypes.Array(numba.types.scalars.Float(name="float64"), 1, 'C')
            twoD_arg = numba.types.npytypes.Array(numba.types.scalars.Float(name="float64"), 2, 'C')
            args = (oneD_arg, twoD_arg, oneD_arg, types.int64)
            #print("args = ", args)
            tp = TestPipeline(typingctx, targetctx, args, test_ir)

            numba.rewrites.rewrite_registry.apply('before-inference', tp, tp.func_ir)

            tp.typemap, tp.return_type, tp.calltypes = compiler.type_inference_stage(tp.typingctx, tp.func_ir, tp.args, None)
            #print("typemap = ", tp.typemap)
            #print("return_type = ", tp.return_type)

            type_annotation = type_annotations.TypeAnnotation(
                func_ir=tp.func_ir,
                typemap=tp.typemap,
                calltypes=tp.calltypes,
                lifted=(),
                lifted_from=None,
                args=tp.args,
                return_type=tp.return_type,
                html_output=config.HTML)

            numba.rewrites.rewrite_registry.apply('after-inference', tp, tp.func_ir)

            parfor_pass = numba.parfor.ParforPass(tp.func_ir, tp.typemap, tp.calltypes, tp.return_type)
            parfor_pass.run()
            #print(tp.func_ir.dump())
            #print(countParfors(tp.func_ir) == 1)
            self.assertTrue(countParfors(test_ir) == 1)

    @unittest.skipIf(not (sys.platform.startswith('win32')
                          and sys.version_info[:2] == (2, 7)),
                    "Only impacts Windows with Python 2.7")
    def test_windows_py27_combination_raises(self):
        """
        This test is in place until issues with the 'parallel'
        target on Windows with Python 2.7 are fixed.
        """
        
        @njit(parallel=True)
        def ddot(a, v):
            return np.dot(a, v)

        A = np.linspace(0, 1, 20).reshape(2, 10)
        v = np.linspace(2, 1, 10)
        with self.assertRaises(RuntimeError) as raised:
            ddot(A, v)
        msg = ("The 'parallel' target is not currently supported on "
            "Windows operating systems when using Python 2.7.")
        self.assertIn(msg, str(raised.exception))

    def test_kde(self):
        n = 128
        X = np.random.ranf(n)
        cfunc = njit(parallel=True)(kde_example)
        expected = kde_example(X)
        output = cfunc(X)
        np.testing.assert_almost_equal(expected, output, decimal=1)
        self.assertIn('@do_scheduling', cfunc.inspect_llvm(cfunc.signatures[0]))

    def test_bulk_cases(self):


        def case01(v1, v2, m1, m2):
            return np.ones(())

        def case02(v1, v2, m1, m2):
            return np.ones((1,))

        def case03(v1, v2, m1, m2):
            return np.ones((-1, 2))

        def case04(v1, v2, m1, m2):
            return np.ones(((1, 2)))

        def case05(v1, v2, m1, m2):
            return np.ones(((1, 2), (3,)))

        def case06(v1, v2, m1, m2):
            return np.ones(((1., 2.)))

        def case07(v1, v2, m1, m2):
            return np.ones(1)

        def case08(v1, v2, m1, m2):
            return np.ones([1])

        def case09(v1, v2, m1, m2):
            return np.ones(([x for x in range(3)]))

        def case10(v1, v2, m1, m2):
            return np.ones((1, 2), dtype=np.complex128)

        def case11(v1, v2, m1, m2):
            return np.ones((1, 2)) + np.ones((1, 2))

        def case12(v1, v2, m1, m2):
            return np.ones((1, 2)) + np.ones((1, 2))

        def case13(v1, v2, m1, m2):
            return np.ones((1, 1))

        def case14(v1, v2, m1, m2):
            return np.ones((0, 0))

        def case15(v1, v2, m1, m2):
            return np.ones((10, 10)) + 1.

        def case16(v1, v2, m1, m2):
            return np.ones((10, 10)) + np.complex128(1.)

        def case17(v1, v2, m1, m2):
            return np.complex128(1.)

        def case18(v1, v2, m1, m2):
            return np.ones((10, 10))[0::20]

        def case19(v1, v2, m1, m2):
            return v1 + v2

        def case20(v1, v2, m1, m2):
            return m1 + m2

        def case21(v1, v2, m1, m2):
            return m1 + v1

        def case22(v1, v2, m1, m2):
            return m1 + v2

        def case23(v1, v2, m1, m2):
            return m1 + v2

        def case24(v1, v2, m1, m2):
            return m1 + np.linalg.svd(m2)[0][:-1, :]

        def case25(v1, v2, m1, m2):
            return np.dot(m1, v1)

        def case26(v1, v2, m1, m2):
            return np.dot(m1, v2)

        def case27(v1, v2, m1, m2):
            return np.dot(m2, v1)

        def case28(v1, v2, m1, m2):
            return np.dot(m1, m2)

        def case29(v1, v2, m1, m2):
            return np.dot(v1, v1)

        def case30(v1, v2, m1, m2):
            return np.sum(m1 + m2.T)

        def case31(v1, v2, m1, m2):
            return np.sum(v1 + v1)

        def case32(v1, v2, m1, m2):
            x = 2 * v1
            y = 2 * v1
            return 4 * np.sum(x**2 + y**2 < 1) / 10

        m = np.reshape(np.arange(12.), (3, 4))
        default_kwargs = {'v1':np.arange(3.), 'v2':np.arange(4.), 'm1':m, 'm2':m.T}


        cm = re.compile('^case[0-9]+$')
        lv = dict(locals())
        cases = [lv[x] for x in sorted([x for x in lv if cm.match(x)])]

        for case in cases:
            print("\n")
            should_have_failed = False
            njit_failed = False
            parfors_failed = False
            got = None

            pyfunc = case
            try:
                expected = pyfunc(**default_kwargs)
            except Exception:
                should_have_failed = True


            try:
                cfunc = njit(pyfunc)
                cfunc(**default_kwargs)
            except Exception as e:
                njit_failed = True

            try:
                pfunc = njit(parallel=True)(pyfunc)
                got = pfunc(**default_kwargs)
                if not should_have_failed:
                    try:
                        np.testing.assert_almost_equal(got, expected)
                        try:
                            assert ('@do_scheduling' in pfunc.inspect_llvm(pfunc.signatures[0]))
                        except AssertionError as raised:
                            parfors_failed = True
                    except Exception as raised:
                        if not njit_failed:
                            print("Fail. %s: %s\n" % (case, raised))
            except Exception as raised:
                if njit_failed:
                    print("Pass (with njit fail). %s\n" % case)
                    continue
                if should_have_failed:
                    print("Pass (with py fail). %s\n" % case)
                    continue
                print("Fail. %s: %s\n" % (case, raised))
                continue

            if parfors_failed:
                print("Pass (with parfors fail). %s\n" % case)
            else:
                print("Pass. %s\n" % case)

if __name__ == "__main__":
    unittest.main()
