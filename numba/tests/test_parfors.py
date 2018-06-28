#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

from __future__ import print_function, division, absolute_import

from math import sqrt
import numbers
import re
import sys
import platform
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
from numba.ir_utils import (find_callname, guard, build_definitions,
                            get_definition, is_getitem, is_setitem,
                            index_var_of_get_setitem)
from numba import ir
from numba.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.compiler import compile_isolated, Flags
from numba.bytecode import ByteCodeIter
from .support import tag, override_env_config
from .matmul_usecase import needs_blas
from .test_linalg import needs_lapack

# for decorating tests, marking that Windows with Python 2.7 is not supported
_windows_py27 = (sys.platform.startswith('win32') and
                 sys.version_info[:2] == (2, 7))
_32bit = sys.maxsize <= 2 ** 32
_reason = 'parfors not supported'
skip_unsupported = unittest.skipIf(_32bit or _windows_py27, _reason)
test_disabled = unittest.skipIf(True, 'Test disabled')
_lnx_reason = 'linux only test'
linux_only = unittest.skipIf(not sys.platform.startswith('linux'), _lnx_reason)
x86_only = unittest.skipIf(platform.machine() not in ('i386', 'x86_64'), 'x86 only test')

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

        # flags for njit(parallel=True, fastmath=True)
        self.fast_pflags = Flags()
        self.fast_pflags.set('auto_parallel', cpu.ParallelOptions(True))
        self.fast_pflags.set('nrt')
        self.fast_pflags.set('fastmath')
        super(TestParforsBase, self).__init__(*args)

    def _compile_this(self, func, sig, flags):
        return compile_isolated(func, sig, flags=flags)

    def compile_parallel(self, func, sig):
        return self._compile_this(func, sig, flags=self.pflags)

    def compile_parallel_fastmath(self, func, sig):
        return self._compile_this(func, sig, flags=self.fast_pflags)

    def compile_njit(self, func, sig):
        return self._compile_this(func, sig, flags=self.cflags)

    def compile_all(self, pyfunc, *args, **kwargs):
        sig = tuple([numba.typeof(x) for x in args])

        # compile the prange injected function
        cpfunc = self.compile_parallel(pyfunc, sig)

        # compile a standard njit of the original function
        cfunc = self.compile_njit(pyfunc, sig)

        return cfunc, cpfunc

    def check_parfors_vs_others(self, pyfunc, cfunc, cpfunc, *args, **kwargs):
        """
        Checks python, njit and parfor impls produce the same result.

        Arguments:
            pyfunc - the python function to test
            cfunc - CompilerResult from njit of pyfunc
            cpfunc - CompilerResult from njit(parallel=True) of pyfunc
            args - arguments for the function being tested
        Keyword Arguments:
            scheduler_type - 'signed', 'unsigned' or None, default is None.
                           Supply in cases where the presence of a specific
                           scheduler is to be asserted.
            fastmath_pcres - a fastmath parallel compile result, if supplied
                             will be run to make sure the result is correct
            Remaining kwargs are passed to np.testing.assert_almost_equal
        """
        scheduler_type = kwargs.pop('scheduler_type', None)
        check_fastmath = kwargs.pop('check_fastmath', None)
        fastmath_pcres = kwargs.pop('fastmath_pcres', None)

        def copy_args(*args):
            if not args:
                return tuple()
            new_args = []
            for x in args:
                if isinstance(x, np.ndarray):
                    new_args.append(x.copy('k'))
                elif isinstance(x, np.number):
                    new_args.append(x.copy())
                elif isinstance(x, numbers.Number):
                    new_args.append(x)
                else:
                    raise ValueError('Unsupported argument type encountered')
            return tuple(new_args)

        # python result
        py_expected = pyfunc(*copy_args(*args))

        # njit result
        njit_output = cfunc.entry_point(*copy_args(*args))

        # parfor result
        parfor_output = cpfunc.entry_point(*copy_args(*args))

        np.testing.assert_almost_equal(njit_output, py_expected, **kwargs)
        np.testing.assert_almost_equal(parfor_output, py_expected, **kwargs)

        self.check_scheduling(cpfunc, scheduler_type)

        # if requested check fastmath variant
        if fastmath_pcres is not None:
            parfor_fastmath_output = fastmath_pcres.entry_point(*copy_args(*args))
            np.testing.assert_almost_equal(parfor_fastmath_output, py_expected,
                                           **kwargs)


    def check_scheduling(self, cres, scheduler_type):
        # make sure parfor set up scheduling
        scheduler_str = '@do_scheduling'
        if scheduler_type is not None:
            if scheduler_type in ['signed', 'unsigned']:
                scheduler_str += '_' + scheduler_type
            else:
                msg = "Unknown scheduler_type specified: %s"
                raise ValueError(msg % scheduler_type)

        self.assertIn(scheduler_str, cres.library.get_llvm_str())

    def _filter_mod(self, mod, magicstr, checkstr=None):
        """ helper function to filter out modules by name"""
        filt = [x for x in mod if magicstr in x.name]
        if checkstr is not None:
            for x in filt:
                assert checkstr in str(x)
        return filt

    def _get_gufunc_modules(self, cres, magicstr, checkstr=None):
        """ gets the gufunc LLVM Modules"""
        _modules = [x for x in cres.library._codegen._engine._ee._modules]
        return self._filter_mod(_modules, magicstr, checkstr=checkstr)

    def _get_gufunc_info(self, cres, fn):
        """ helper for gufunc IR/asm generation"""
        # get the gufunc modules
        magicstr = '__numba_parfor_gufunc'
        gufunc_mods = self._get_gufunc_modules(cres, magicstr)
        x = dict()
        for mod in gufunc_mods:
            x[mod.name] = fn(mod)
        return x

    def _get_gufunc_ir(self, cres):
        """
        Returns the IR of the gufuncs used as parfor kernels
        as a dict mapping the gufunc name to its IR.

        Arguments:
         cres - a CompileResult from `njit(parallel=True, ...)`
        """
        return self._get_gufunc_info(cres, str)

    def _get_gufunc_asm(self, cres):
        """
        Returns the assembly of the gufuncs used as parfor kernels
        as a dict mapping the gufunc name to its assembly.

        Arguments:
         cres - a CompileResult from `njit(parallel=True, ...)`
        """
        tm = cres.library._codegen._tm
        def emit_asm(mod):
            return str(tm.emit_assembly(mod))
        return self._get_gufunc_info(cres, emit_asm)

    def assert_fastmath(self, pyfunc, sig):
        """
        Asserts that the fastmath flag has some effect in that suitable
        instructions are now labelled as `fast`. Whether LLVM can actually do
        anything to optimise better now the derestrictions are supplied is
        another matter!

        Arguments:
         pyfunc - a function that contains operations with parallel semantics
         sig - the type signature of pyfunc
        """

        cres = self.compile_parallel_fastmath(pyfunc, sig)
        _ir = self._get_gufunc_ir(cres)

        def _get_fast_instructions(ir):
            splitted = ir.splitlines()
            fast_inst = []
            for x in splitted:
                if 'fast' in x:
                    fast_inst.append(x)
            return fast_inst

        def _assert_fast(instrs):
            ops = ('fadd', 'fsub', 'fmul', 'fdiv', 'frem', 'fcmp')
            for inst in instrs:
                count = 0
                for op in ops:
                    match = op + ' fast'
                    if match in inst:
                        count += 1
                self.assertTrue(count > 0)

        for name, guir in _ir.items():
            inst = _get_fast_instructions(guir)
            _assert_fast(inst)


def blackscholes_impl(sptprice, strike, rate, volatility, timev):
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


def lr_impl(Y, X, w, iterations):
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

def get_optimized_numba_ir(test_func, args, **kws):
    typingctx = typing.Context()
    targetctx = cpu.CPUContext(typingctx)
    test_ir = compiler.run_frontend(test_func)
    if kws:
        options = cpu.ParallelOptions(kws)
    else:
        options = cpu.ParallelOptions(True)

    tp = TestPipeline(typingctx, targetctx, args, test_ir)

    with cpu_target.nested_context(typingctx, targetctx):
        typingctx.refresh()
        targetctx.refresh()

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

        flags = compiler.Flags()
        parfor_pass = numba.parfor.ParforPass(
            tp.func_ir, tp.typemap, tp.calltypes, tp.return_type,
            tp.typingctx, options, flags)
        parfor_pass.run()
        test_ir._definitions = build_definitions(test_ir.blocks)

    return test_ir, tp

def countParfors(test_func, args, **kws):
    test_ir, tp = get_optimized_numba_ir(test_func, args, **kws)
    ret_count = 0

    for label, block in test_ir.blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, numba.parfor.Parfor):
                ret_count += 1

    return ret_count


def countArrays(test_func, args, **kws):
    test_ir, tp = get_optimized_numba_ir(test_func, args, **kws)
    return _count_arrays_inner(test_ir.blocks, tp.typemap)

def _count_arrays_inner(blocks, typemap):
    ret_count = 0
    arr_set = set()

    for label, block in blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, numba.parfor.Parfor):
                parfor_blocks = inst.loop_body.copy()
                parfor_blocks[0] = inst.init_block
                ret_count += _count_arrays_inner(parfor_blocks, typemap)
            if (isinstance(inst, ir.Assign)
                    and isinstance(typemap[inst.target.name],
                                    types.ArrayCompatible)):
                arr_set.add(inst.target.name)

    ret_count += len(arr_set)
    return ret_count

def countArrayAllocs(test_func, args, **kws):
    test_ir, tp = get_optimized_numba_ir(test_func, args, **kws)
    ret_count = 0

    for block in test_ir.blocks.values():
        ret_count += _count_array_allocs_inner(test_ir, block)

    return ret_count

def _count_array_allocs_inner(func_ir, block):
    ret_count = 0
    for inst in block.body:
        if isinstance(inst, numba.parfor.Parfor):
            ret_count += _count_array_allocs_inner(func_ir, inst.init_block)
            for b in inst.loop_body.values():
                ret_count += _count_array_allocs_inner(func_ir, b)

        if (isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr)
                and inst.value.op == 'call'
                and (guard(find_callname, func_ir, inst.value) == ('empty', 'numpy')
                or guard(find_callname, func_ir, inst.value)
                    == ('empty_inferred', 'numba.unsafe.ndarray'))):
            ret_count += 1

    return ret_count

def countNonParforArrayAccesses(test_func, args, **kws):
    test_ir, tp = get_optimized_numba_ir(test_func, args, **kws)
    return _count_non_parfor_array_accesses_inner(test_ir, test_ir.blocks, tp.typemap)

def _count_non_parfor_array_accesses_inner(f_ir, blocks, typemap, parfor_indices=None):
    ret_count = 0
    if parfor_indices is None:
        parfor_indices = set()

    for label, block in blocks.items():
        for stmt in block.body:
            if isinstance(stmt, numba.parfor.Parfor):
                parfor_indices.add(stmt.index_var.name)
                parfor_blocks = stmt.loop_body.copy()
                parfor_blocks[0] = stmt.init_block
                ret_count += _count_non_parfor_array_accesses_inner(
                    f_ir, parfor_blocks, typemap, parfor_indices)

            # getitem
            if (is_getitem(stmt) and isinstance(typemap[stmt.value.value.name],
                        types.ArrayCompatible) and not _uses_indices(
                        f_ir, index_var_of_get_setitem(stmt), parfor_indices)):
                ret_count += 1

            # setitem
            if (is_setitem(stmt) and isinstance(typemap[stmt.target.name],
                    types.ArrayCompatible) and not _uses_indices(
                    f_ir, index_var_of_get_setitem(stmt), parfor_indices)):
                ret_count += 1

    return ret_count

def _uses_indices(f_ir, index, index_set):
    if index.name in index_set:
        return True

    ind_def = guard(get_definition, f_ir, index)
    if isinstance(ind_def, ir.Expr) and ind_def.op == 'build_tuple':
        varnames = set(v.name for v in ind_def.items)
        return len(varnames & index_set) != 0

    return False


class TestPipeline(object):
    def __init__(self, typingctx, targetctx, args, test_ir):
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
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

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
        self.assertTrue(countArrays(test_impl, (types.intp,)) == 0)

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
        self.assertTrue(countArrays(test_impl, (types.intp,)) == 0)

    @skip_unsupported
    @tag('important')
    def test_blackscholes(self):
        # blackscholes takes 5 1D float array args
        args = (numba.float64[:], ) * 5
        self.assertTrue(countParfors(blackscholes_impl, args) == 1)

    @skip_unsupported
    @needs_blas
    @tag('important')
    def test_logistic_regression(self):
        args = (numba.float64[:], numba.float64[:,:], numba.float64[:],
                numba.int64)
        self.assertTrue(countParfors(lr_impl, args) == 1)
        self.assertTrue(countArrayAllocs(lr_impl, args) == 1)

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
        arg_typs = (types.Array(types.float64, 2, 'C'), types.intp, types.intp,
                    types.Array(types.float64, 2, 'C'))
        self.assertTrue(
            countNonParforArrayAccesses(test_kmeans_example, arg_typs) == 0)

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
    def test_simple24(self):
        def test_impl():
            n = 20
            A = np.ones((n, n))
            b = np.arange(n)
            return np.sum(A[:, b])
        self.check(test_impl)

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


    @skip_unsupported
    def test_parfor_array_access1(self):
        # signed index of the prange generated by sum() should be replaced
        # resulting in array A to be eliminated (see issue #2846)
        def test_impl(n):
            A = np.ones(n)
            return A.sum()

        n = 211
        self.check(test_impl, n)
        self.assertEqual(countArrays(test_impl, (types.intp,)), 0)

    @skip_unsupported
    def test_parfor_array_access2(self):
        # in this test, the prange index has the same name (i) in two loops
        # thus, i has multiple definitions and is harder to replace
        def test_impl(n):
            A = np.ones(n)
            m = 0
            n = 0
            for i in numba.prange(len(A)):
                m += A[i]

            for i in numba.prange(len(A)):
                if m == n:  # access in another block
                    n += A[i]

            return m + n

        n = 211
        self.check(test_impl, n)
        self.assertEqual(countNonParforArrayAccesses(test_impl, (types.intp,)), 0)

    @skip_unsupported
    def test_parfor_array_access3(self):
        def test_impl(n):
            A = np.ones(n, np.int64)
            m = 0
            for i in numba.prange(len(A)):
                m += A[i]
                if m==2:
                    i = m

        n = 211
        with self.assertRaises(ValueError) as raises:
            self.check(test_impl, n)
        self.assertIn("Overwrite of parallel loop index", str(raises.exception))

    @skip_unsupported
    @needs_blas
    def test_parfor_array_access4(self):
        # in this test, one index of a multi-dim access should be replaced
        # np.dot parallel implementation produces this case
        def test_impl(A, b):
            return np.dot(A, b)

        n = 211
        d = 4
        A = np.random.ranf((n, d))
        b = np.random.ranf(d)
        self.check(test_impl, A, b)
        # make sure the parfor index is replaced in build_tuple of access to A
        test_ir, tp = get_optimized_numba_ir(
            test_impl, (types.Array(types.float64, 2, 'C'),
                        types.Array(types.float64, 1, 'C')))
        # this code should have one basic block after optimization
        self.assertTrue(len(test_ir.blocks) == 1 and 0 in test_ir.blocks)
        block = test_ir.blocks[0]
        parfor_found = False
        parfor = None
        for stmt in block.body:
            if isinstance(stmt, numba.parfor.Parfor):
                parfor_found = True
                parfor = stmt

        self.assertTrue(parfor_found)
        build_tuple_found = False
        # there should be only one build_tuple
        for bl in parfor.loop_body.values():
            for stmt in bl.body:
                if (isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op == 'build_tuple'):
                    build_tuple_found = True
                    self.assertTrue(parfor.index_var in stmt.value.items)

        self.assertTrue(build_tuple_found)

    @skip_unsupported
    def test_parfor_array_access5(self):
        # one dim is slice in multi-dim access
        def test_impl(n):
            X = np.ones((n, 3))
            y = 0
            for i in numba.prange(n):
                y += X[i,:].sum()
            return y

        n = 211
        self.check(test_impl, n)
        self.assertEqual(countNonParforArrayAccesses(test_impl, (types.intp,)), 0)

    @skip_unsupported
    @needs_blas
    def test_parfor_generate_fuse(self):
        # issue #2857
        def test_impl(N, D):
            w = np.ones(D)
            X = np.ones((N, D))
            Y = np.ones(N)
            for i in range(3):
                B = (-Y * np.dot(X, w))

            return B

        n = 211
        d = 3
        self.check(test_impl, n, d)
        self.assertEqual(countArrayAllocs(test_impl, (types.intp, types.intp)), 4)
        self.assertEqual(countParfors(test_impl, (types.intp, types.intp)), 4)

    @skip_unsupported
    def test_ufunc_expr(self):
        # issue #2885
        def test_impl(A, B):
            return np.bitwise_and(A, B)

        A = np.ones(3, np.uint8)
        B = np.ones(3, np.uint8)
        B[1] = 0
        self.check(test_impl, A, B)

    @skip_unsupported
    def test_find_callname_intrinsic(self):
        def test_impl(n):
            A = unsafe_empty((n,))
            for i in range(n):
                A[i] = i + 2.0
            return A

        # the unsafe allocation should be found even though it is imported
        # as a different name
        self.assertEqual(countArrayAllocs(test_impl, (types.intp,)), 1)


class TestPrangeBase(TestParforsBase):

    def __init__(self, *args):
        TestParforsBase.__init__(self, *args)

    def generate_prange_func(self, pyfunc, patch_instance):
        """
        This function does the actual code augmentation to enable the explicit
        testing of `prange` calls in place of `range`.
        """
        pyfunc_code = pyfunc.__code__

        prange_names = list(pyfunc_code.co_names)

        if patch_instance is None:
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

        return pfunc

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
         scheduler_type - 'signed', 'unsigned' or None, default is None.
                           Supply in cases where the presence of a specific
                           scheduler is to be asserted.
         check_fastmath - if True then a check will be performed to ensure the
                          IR contains instructions labelled with 'fast'
         check_fastmath_result - if True then a check will be performed to
                                 ensure the result of running with fastmath
                                 on matches that of the pyfunc
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
        patch_instance = kwargs.pop('patch_instance', None)
        check_fastmath = kwargs.pop('check_fastmath', False)
        check_fastmath_result = kwargs.pop('check_fastmath_result', False)

        pfunc = self.generate_prange_func(pyfunc, patch_instance)

        # Compile functions
        # compile a standard njit of the original function
        sig = tuple([numba.typeof(x) for x in args])
        cfunc = self.compile_njit(pyfunc, sig)

        # compile the prange injected function
        cpfunc = self.compile_parallel(pfunc, sig)

        # if check_fastmath is True then check fast instructions
        if check_fastmath:
            self.assert_fastmath(pfunc, sig)

        # if check_fastmath_result is True then compile a function
        # so that the parfors checker can assert the result is ok.
        if check_fastmath_result:
            fastcpfunc = self.compile_parallel_fastmath(pfunc, sig)
            kwargs = dict({'fastmath_pcres': fastcpfunc}, **kwargs)

        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)


class TestPrange(TestPrangeBase):
    """ Tests Prange """

    @skip_unsupported
    def test_prange01(self):
        def test_impl():
            n = 4
            A = np.zeros(n)
            for i in range(n):
                A[i] = 2.0 * i
            return A
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_unsupported
    def test_prange02(self):
        def test_impl():
            n = 4
            A = np.zeros(n - 1)
            for i in range(1, n):
                A[i - 1] = 2.0 * i
            return A
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_unsupported
    def test_prange03(self):
        def test_impl():
            s = 0
            for i in range(10):
                s += 2
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

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
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_unsupported
    def test_prange05(self):
        def test_impl():
            n = 4
            A = np.ones((n), dtype=np.float64)
            s = 0
            for i in range(1, n - 1, 1):
                s += A[i]
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_unsupported
    def test_prange06(self):
        def test_impl():
            n = 4
            A = np.ones((n), dtype=np.float64)
            s = 0
            for i in range(1, 1, 1):
                s += A[i]
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_unsupported
    def test_prange07(self):
        def test_impl():
            n = 4
            A = np.ones((n), dtype=np.float64)
            s = 0
            for i in range(n, 1):
                s += A[i]
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

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
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

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
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

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
        self.prange_tester(test_impl, patch_instance=[1],
                           scheduler_type='unsigned',
                           check_fastmath=True)

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
        self.prange_tester(test_impl, patch_instance=[0],
                           scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_unsupported
    @unittest.skip("list append is not thread-safe yet (#2391, #2408)")
    def test_prange11(self):
        def test_impl():
            n = 4
            return [np.sin(j) for j in range(n)]
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_unsupported
    def test_prange12(self):
        def test_impl():
            acc = 0
            n = 4
            X = np.ones(n)
            for i in range(-len(X)):
                acc += X[i]
            return acc
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_unsupported
    def test_prange13(self):
        def test_impl(n):
            acc = 0
            for i in range(n):
                acc += 1
            return acc
        self.prange_tester(test_impl, np.int32(4), scheduler_type='unsigned',
                           check_fastmath=True)

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
        self.prange_tester(test_impl, np.random.ranf(4),
                           scheduler_type='unsigned',
                           check_fastmath=True)

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
        self.prange_tester(test_impl, 1024, scheduler_type='unsigned',
                           check_fastmath=True)

    # Tests for negative ranges
    @skip_unsupported
    def test_prange16(self):
        def test_impl(N):
            acc = 0
            for i in range(-N, N):
                acc += 2
            return acc
        self.prange_tester(test_impl, 1024, scheduler_type='signed',
                           check_fastmath=True)

    @skip_unsupported
    def test_prange17(self):
        def test_impl(N):
            acc = 0
            X = np.ones(N)
            for i in range(-N, N):
                acc += X[i]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed',
                           check_fastmath=True)

    @skip_unsupported
    def test_prange18(self):
        def test_impl(N):
            acc = 0
            X = np.ones(N)
            for i in range(-N, 5):
                acc -= X[i]
                for j in range(-4, N):
                    acc += X[j]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed',
                           check_fastmath=True)

    @skip_unsupported
    def test_prange19(self):
        def test_impl(N):
            acc = 0
            M = N + 4
            X = np.ones((N, M))
            for i in range(-N, N):
                for j in range(-M, M):
                    acc += X[i, j]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed',
                           check_fastmath=True)

    @skip_unsupported
    def test_prange20(self):
        def test_impl(N):
            acc = 0
            X = np.ones(N)
            for i in range(-1, N):
                acc += X[i]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed',
                           check_fastmath=True)

    @skip_unsupported
    def test_prange21(self):
        def test_impl(N):
            acc = 0
            for i in range(-3, -1):
                acc += 3
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed',
                           check_fastmath=True)

    @skip_unsupported
    def test_prange22(self):
        def test_impl():
            a = 0
            b = 3
            A = np.empty(4)
            for i in range(-2, 2):
                if i == a:
                    A[i] = b
                elif i < 1:
                    A[i] = -1
                else:
                    A[i] = 7
            return A
        self.prange_tester(test_impl, scheduler_type='signed',
                           check_fastmath=True, check_fastmath_result=True)

    @skip_unsupported
    def test_prange23(self):
        # test non-contig input
        def test_impl(A):
            for i in range(len(A)):
                A[i] = i
            return A
        A = np.zeros(32)[::2]
        self.prange_tester(test_impl, A, scheduler_type='unsigned',
                           check_fastmath=True, check_fastmath_result=True)

    @skip_unsupported
    def test_prange24(self):
        # test non-contig input, signed range
        def test_impl(A):
            for i in range(-len(A), 0):
                A[i] = i
            return A
        A = np.zeros(32)[::2]
        self.prange_tester(test_impl, A, scheduler_type='signed',
                           check_fastmath=True, check_fastmath_result=True)

    # should this work?
    @skip_unsupported
    def test_prange25(self):
        def test_impl(A):
            B = A[::3]
            for i in range(len(B)):
                B[i] = i
            return A
        A = np.zeros(32)[::2]
        self.prange_tester(test_impl, A, scheduler_type='unsigned',
                           check_fastmath=True, check_fastmath_result=True)

#    @skip_unsupported
    @test_disabled
    def test_check_error_model(self):
        def test_impl():
            n = 32
            A = np.zeros(n)
            for i in range(n):
                A[i] = 1 / i # div-by-zero when i = 0
            return A

        with self.assertRaises(ZeroDivisionError) as raises:
            test_impl()

        # compile parallel functions
        pfunc = self.generate_prange_func(test_impl, None)
        pcres = self.compile_parallel(pfunc, ())
        pfcres = self.compile_parallel_fastmath(pfunc, ())

        # should raise
        with self.assertRaises(ZeroDivisionError) as raises:
            pcres.entry_point()

        # should not raise
        result = pfcres.entry_point()
        self.assertEqual(result[0], np.inf)


    @skip_unsupported
    def test_check_alias_analysis(self):
        # check alias analysis reports ok
        def test_impl(A):
            for i in range(len(A)):
                B = A[i]
                B[:] = 1
            return A
        A = np.zeros(32).reshape(4, 8)
        self.prange_tester(test_impl, A, scheduler_type='unsigned',
                           check_fastmath=True, check_fastmath_result=True)
        pfunc = self.generate_prange_func(test_impl, None)
        sig = tuple([numba.typeof(A)])
        cres = self.compile_parallel_fastmath(pfunc, sig)
        _ir = self._get_gufunc_ir(cres)
        for k, v in _ir.items():
            for line in v.splitlines():
                # get the fn definition line
                if 'define' in line and k in line:
                    # there should only be 2x noalias, one on each of the first
                    # 2 args (retptr, excinfo).
                    # Note: used to be 3x no noalias, but env arg is dropped.
                    self.assertEqual(line.count('noalias'), 2)
                    break

    @skip_unsupported
    def test_prange_raises_invalid_step_size(self):
        def test_impl(N):
            acc = 0
            for i in range(0, N, 2):
                acc += 2
            return acc

        with self.assertRaises(NotImplementedError) as raises:
            self.prange_tester(test_impl, 1024)
        msg = 'Only constant step size of 1 is supported for prange'
        self.assertIn(msg, str(raises.exception))

    @skip_unsupported
    def test_prange_fastmath_check_works(self):
        # this function will benefit from `fastmath`, the div will
        # get optimised to a multiply by reciprocal and the accumulator
        # then becomes an fmadd: A = A + i * 0.5
        def test_impl():
            n = 128
            A = 0
            for i in range(n):
                A += i / 2.0
            return A
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)
        pfunc = self.generate_prange_func(test_impl, None)
        cres = self.compile_parallel_fastmath(pfunc, ())
        ir = self._get_gufunc_ir(cres)
        _id = '%[A-Z]?.[0-9]+[.]?[i]?'
        recipr_str = '\s+%s = fmul fast double %s, 5.000000e-01'
        reciprocal_inst = re.compile(recipr_str % (_id, _id))
        fadd_inst = re.compile('\s+%s = fadd fast double %s, %s'
                               % (_id, _id, _id))
        # check there is something like:
        #  %.329 = fmul fast double %.325, 5.000000e-01
        #  %.337 = fadd fast double %A.07, %.329
        for name, kernel in ir.items():
            splitted = kernel.splitlines()
            for i, x in enumerate(splitted):
                if reciprocal_inst.match(x):
                    break
            self.assertTrue(fadd_inst.match(splitted[i + 1]))

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


@x86_only
class TestParforsVectorizer(TestPrangeBase):

    # env mutating test
    _numba_parallel_test_ = False

    def get_gufunc_asm(self, func, schedule_type, *args, **kwargs):

        fastmath = kwargs.pop('fastmath', False)
        nthreads = kwargs.pop('nthreads', 2)
        cpu_name = kwargs.pop('cpu_name', 'skylake-avx512')
        assertions = kwargs.pop('assertions', True)

        env_opts = {'NUMBA_CPU_NAME': cpu_name,
                    'NUMBA_CPU_FEATURES': '',
                    'NUMBA_NUM_THREADS': str(nthreads)
                    }

        overrides = []
        for k, v in env_opts.items():
            overrides.append(override_env_config(k, v))

        with overrides[0], overrides[1], overrides[2]:
            sig = tuple([numba.typeof(x) for x in args])
            pfunc_vectorizable = self.generate_prange_func(func, None)
            if fastmath == True:
                cres = self.compile_parallel_fastmath(pfunc_vectorizable, sig)
            else:
                cres = self.compile_parallel(pfunc_vectorizable, sig)

            # get the gufunc asm
            asm = self._get_gufunc_asm(cres)

            if assertions:
                schedty = re.compile('call\s+\w+\*\s+@do_scheduling_(\w+)\(')
                matches = schedty.findall(cres.library.get_llvm_str())
                self.assertEqual(len(matches), 2) # 1x decl, 1x call
                self.assertEqual(matches[0], matches[1])
                self.assertTrue(asm != {})

        return asm

    # this is a common match pattern for something like:
    # \n\tvsqrtpd\t-192(%rbx,%rsi,8), %zmm0\n
    # to check vsqrtpd operates on zmm
    match_vsqrtpd_on_zmm = re.compile('\n\s+vsqrtpd\s+.*zmm.*\n')

    @linux_only
    def test_vectorizer_fastmath_asm(self):
        """ This checks that if fastmath is set and the underlying hardware
        is suitable, and the function supplied is amenable to fastmath based
        vectorization, that the vectorizer actually runs.
        """

        # This function will benefit from `fastmath` if run on a suitable
        # target. The vectorizer should unwind the loop and generate
        # packed dtype=double add and sqrt instructions.
        def will_vectorize(A):
            n = len(A)
            acc = 0
            for i in range(n):
                acc += np.sqrt(i)
            return acc

        arg = np.zeros(10)

        fast_asm = self.get_gufunc_asm(will_vectorize, 'unsigned', arg,
                                       fastmath=True)
        slow_asm = self.get_gufunc_asm(will_vectorize, 'unsigned', arg,
                                       fastmath=False)

        for v in fast_asm.values():
            # should unwind and call vector sqrt then vector add
            # all on packed doubles using zmm's
            self.assertTrue('vaddpd' in v)
            self.assertTrue('vsqrtpd' in v)
            self.assertTrue('zmm' in v)
            # make sure vsqrtpd operates on zmm
            self.assertTrue(len(self.match_vsqrtpd_on_zmm.findall(v)) > 1)

        for v in slow_asm.values():
            # vector variants should not be present
            self.assertTrue('vaddpd' not in v)
            self.assertTrue('vsqrtpd' not in v)
            # check scalar variant is present
            self.assertTrue('vsqrtsd' in v)
            self.assertTrue('vaddsd' in v)
            # check no zmm addressing is present
            self.assertTrue('zmm' not in v)

    @linux_only
    def test_unsigned_refusal_to_vectorize(self):
        """ This checks that if fastmath is set and the underlying hardware
        is suitable, and the function supplied is amenable to fastmath based
        vectorization, that the vectorizer actually runs.
        """

        def will_not_vectorize(A):
            n = len(A)
            for i in range(-n, 0):
                A[i] = np.sqrt(A[i])
            return A

        def will_vectorize(A):
            n = len(A)
            for i in range(n):
                A[i] = np.sqrt(A[i])
            return A

        arg = np.zeros(10)

        novec_asm = self.get_gufunc_asm(will_not_vectorize, 'signed', arg,
                                        fastmath=True)

        vec_asm = self.get_gufunc_asm(will_vectorize, 'unsigned', arg,
                                      fastmath=True)

        for v in novec_asm.values():
            # vector variant should not be present
            self.assertTrue('vsqrtpd' not in v)
            # check scalar variant is present
            self.assertTrue('vsqrtsd' in v)
            # check no zmm addressing is present
            self.assertTrue('zmm' not in v)

        for v in vec_asm.values():
            # should unwind and call vector sqrt then vector mov
            # all on packed doubles using zmm's
            self.assertTrue('vsqrtpd' in v)
            self.assertTrue('vmovupd' in v)
            self.assertTrue('zmm' in v)
            # make sure vsqrtpd operates on zmm
            self.assertTrue(len(self.match_vsqrtpd_on_zmm.findall(v)) > 1)

    @linux_only
    # needed as 32bit doesn't have equivalent signed/unsigned instruction generation
    # for this function
    @skip_unsupported
    def test_signed_vs_unsigned_vec_asm(self):
        """ This checks vectorization for signed vs unsigned variants of a
        trivial accumulator, the only meaningful difference should be the
        presence of signed vs. unsigned unpack instructions (for the
        induction var).
        """
        def signed_variant():
            n = 4096
            A = 0.
            for i in range(-n, 0):
                A += i
            return A

        def unsigned_variant():
            n = 4096
            A = 0.
            for i in range(n):
                A += i
            return A

        signed_asm = self.get_gufunc_asm(signed_variant, 'signed',
                                         fastmath=True)
        unsigned_asm = self.get_gufunc_asm(unsigned_variant, 'unsigned',
                                           fastmath=True)

        def strip_instrs(asm):
            acc = []
            for x in asm.splitlines():
                spd = x.strip()
                # filter out anything that isn't a trivial instruction
                # and anything with the gufunc id as it contains an address
                if spd != '' and not (spd.startswith('.')
                                     or spd.startswith('_')
                                     or spd.startswith('"')
                                     or '__numba_parfor_gufunc' in spd):
                        acc.append(re.sub('[\t]', '', spd))
            return acc

        for k, v in signed_asm.items():
            signed_instr = strip_instrs(v)
            break

        for k, v in unsigned_asm.items():
            unsigned_instr = strip_instrs(v)
            break

        from difflib import SequenceMatcher as sm
        # make sure that the only difference in instruction (if there is a
        # difference) is the char 'u'. For example:
        # vcvtsi2sdq vs. vcvtusi2sdq
        self.assertEqual(len(signed_instr), len(unsigned_instr))
        for a, b in zip(signed_instr, unsigned_instr):
            if a == b:
                continue
            else:
                s = sm(lambda x: x == '\t', a, b)
                ops = s.get_opcodes()
                for op in ops:
                    if op[0] == 'insert':
                        self.assertEqual(b[op[-2]:op[-1]], 'u')


class TestParforsSlice(TestParforsBase):

    def check(self, pyfunc, *args, **kwargs):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

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

#    @skip_unsupported
    @test_disabled
    def test_parfor_slice8(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.transpose()
            b[1:m,1:n] = a[1:m,1:n]
            return b

        self.check(test_impl, np.arange(9).reshape((3,3)))

#    @skip_unsupported
    @test_disabled
    def test_parfor_slice9(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.transpose()
            b[1:n,1:m] = a[:,1:m]
            return b

        self.check(test_impl, np.arange(12).reshape((3,4)))

#    @skip_unsupported
    @test_disabled
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
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

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
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

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
