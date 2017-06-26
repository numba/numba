#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

from __future__ import print_function, division, absolute_import

import sys

import numpy as np

import numba
from numba import unittest_support as unittest
from numba import njit
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
    for i in range(iterations):
        w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X,w))) - 1.0) * Y),X)
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


if __name__ == "__main__":
    unittest.main()
