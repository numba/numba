#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#


from math import sqrt
import numbers
import re
import sys
import dis
import platform
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple

import numba.parfors.parfor
from numba import njit, prange, set_num_threads, get_num_threads
from numba.core import (types, utils, typing, errors, ir, rewrites,
                        typed_passes, inline_closurecall, config, compiler, cpu)
from numba.extending import (overload_method, register_model,
                             typeof_impl, unbox, NativeValue, models)
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
                            get_definition, is_getitem, is_setitem,
                            index_var_of_get_setitem)
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.bytecode import ByteCodeIter
from numba.core.compiler import (compile_isolated, Flags, CompilerBase,
                                 DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
                      override_env_config, linux_only, tag,
                      skip_parfors_unsupported, _32bit, needs_blas,
                      needs_lapack, disabled_test)
import cmath
import unittest


x86_only = unittest.skipIf(platform.machine() not in ('i386', 'x86_64'), 'x86 only test')

_GLOBAL_INT_FOR_TESTING1 = 17
_GLOBAL_INT_FOR_TESTING2 = 5

TestNamedTuple = namedtuple('TestNamedTuple', ('part0', 'part1'))

class TestParforsBase(TestCase):
    """
    Base class for testing parfors.
    Provides functions for compilation and three way comparison between
    python functions, njit'd functions and parfor njit'd functions.
    """

    _numba_parallel_test_ = False

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
        self.fast_pflags.set('fastmath', cpu.FastMathOptions(True))
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
        check_scheduling = kwargs.pop('check_scheduling', True)

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
                elif isinstance(x, tuple):
                    new_args.append(x)
                elif isinstance(x, list):
                    new_args.append(x[:])
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

        self.assertEqual(type(njit_output), type(parfor_output))

        if check_scheduling:
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
                m = re.search(r'\bfast\b', x)  # \b for wholeword
                if m is not None:
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

def example_kmeans_test(A, numCenter, numIter, init_centroids):
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

        inline_pass = inline_closurecall.InlineClosureCallPass(tp.state.func_ir,
                                                               options,
                                                               typed=True)
        inline_pass.run()

        rewrites.rewrite_registry.apply('before-inference', tp.state)

        tp.state.typemap, tp.state.return_type, tp.state.calltypes = \
        typed_passes.type_inference_stage(tp.state.typingctx, tp.state.func_ir,
            tp.state.args, None)

        type_annotations.TypeAnnotation(
            func_ir=tp.state.func_ir,
            typemap=tp.state.typemap,
            calltypes=tp.state.calltypes,
            lifted=(),
            lifted_from=None,
            args=tp.state.args,
            return_type=tp.state.return_type,
            html_output=config.HTML)

        diagnostics = numba.parfors.parfor.ParforDiagnostics()

        preparfor_pass = numba.parfors.parfor.PreParforPass(
            tp.state.func_ir, tp.state.typemap, tp.state.calltypes,
            tp.state.typingctx, options,
            swapped=diagnostics.replaced_fns)
        preparfor_pass.run()

        rewrites.rewrite_registry.apply('after-inference', tp.state)

        flags = compiler.Flags()
        parfor_pass = numba.parfors.parfor.ParforPass(
            tp.state.func_ir, tp.state.typemap, tp.state.calltypes,
            tp.state.return_type, tp.state.typingctx, options, flags,
            diagnostics=diagnostics)
        parfor_pass.run()
        test_ir._definitions = build_definitions(test_ir.blocks)

    return test_ir, tp

def countParfors(test_func, args, **kws):
    test_ir, tp = get_optimized_numba_ir(test_func, args, **kws)
    ret_count = 0

    for label, block in test_ir.blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, numba.parfors.parfor.Parfor):
                ret_count += 1

    return ret_count


def countArrays(test_func, args, **kws):
    test_ir, tp = get_optimized_numba_ir(test_func, args, **kws)
    return _count_arrays_inner(test_ir.blocks, tp.state.typemap)

def get_init_block_size(test_func, args, **kws):
    test_ir, tp = get_optimized_numba_ir(test_func, args, **kws)
    blocks = test_ir.blocks

    ret_count = 0

    for label, block in blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, numba.parfors.parfor.Parfor):
                ret_count += len(inst.init_block.body)

    return ret_count

def _count_arrays_inner(blocks, typemap):
    ret_count = 0
    arr_set = set()

    for label, block in blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, numba.parfors.parfor.Parfor):
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
        if isinstance(inst, numba.parfors.parfor.Parfor):
            ret_count += _count_array_allocs_inner(func_ir, inst.init_block)
            for b in inst.loop_body.values():
                ret_count += _count_array_allocs_inner(func_ir, b)

        if (isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr)
                and inst.value.op == 'call'
                and (guard(find_callname, func_ir, inst.value) == ('empty', 'numpy')
                or guard(find_callname, func_ir, inst.value)
                    == ('empty_inferred', 'numba.np.unsafe.ndarray'))):
            ret_count += 1

    return ret_count

def countNonParforArrayAccesses(test_func, args, **kws):
    test_ir, tp = get_optimized_numba_ir(test_func, args, **kws)
    return _count_non_parfor_array_accesses_inner(test_ir, test_ir.blocks,
                                                  tp.state.typemap)

def _count_non_parfor_array_accesses_inner(f_ir, blocks, typemap, parfor_indices=None):
    ret_count = 0
    if parfor_indices is None:
        parfor_indices = set()

    for label, block in blocks.items():
        for stmt in block.body:
            if isinstance(stmt, numba.parfors.parfor.Parfor):
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
        self.state = compiler.StateDict()
        self.state.typingctx = typingctx
        self.state.targetctx = targetctx
        self.state.args = args
        self.state.func_ir = test_ir
        self.state.typemap = None
        self.state.return_type = None
        self.state.calltypes = None


class TestParfors(TestParforsBase):

    def __init__(self, *args):
        TestParforsBase.__init__(self, *args)
        # these are used in the mass of simple tests
        m = np.reshape(np.arange(12.), (3, 4))
        self.simple_args = [np.arange(3.), np.arange(4.), m, m.T]

    def check(self, pyfunc, *args, **kwargs):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    @skip_parfors_unsupported
    def test_arraymap(self):
        def test_impl(a, x, y):
            return a * x + y

        A = np.linspace(0, 1, 10)
        X = np.linspace(2, 1, 10)
        Y = np.linspace(1, 2, 10)

        self.check(test_impl, A, X, Y)

    @skip_parfors_unsupported
    @needs_blas
    def test_mvdot(self):
        def test_impl(a, v):
            return np.dot(a, v)

        A = np.linspace(0, 1, 20).reshape(2, 10)
        v = np.linspace(2, 1, 10)

        self.check(test_impl, A, v)

    @skip_parfors_unsupported
    def test_0d_broadcast(self):
        def test_impl():
            X = np.array(1)
            Y = np.ones((10, 12))
            return np.sum(X + Y)
        self.check(test_impl)
        self.assertTrue(countParfors(test_impl, ()) == 1)

    @skip_parfors_unsupported
    def test_2d_parfor(self):
        def test_impl():
            X = np.ones((10, 12))
            Y = np.zeros((10, 12))
            return np.sum(X + Y)
        self.check(test_impl)
        self.assertTrue(countParfors(test_impl, ()) == 1)

    @skip_parfors_unsupported
    def test_pi(self):
        def test_impl(n):
            x = 2 * np.random.ranf(n) - 1
            y = 2 * np.random.ranf(n) - 1
            return 4 * np.sum(x**2 + y**2 < 1) / n

        self.check(test_impl, 100000, decimal=1)
        self.assertTrue(countParfors(test_impl, (types.int64, )) == 1)
        self.assertTrue(countArrays(test_impl, (types.intp,)) == 0)

    @skip_parfors_unsupported
    def test_fuse_argmin_argmax_max_min(self):
        for op in [np.argmin, np.argmax, np.min, np.max]:
            def test_impl(n):
                A = np.ones(n)
                C = op(A)
                B = A.sum()
                return B + C
            self.check(test_impl, 256)
            self.assertTrue(countParfors(test_impl, (types.int64, )) == 1)
            self.assertTrue(countArrays(test_impl, (types.intp,)) == 0)

    @skip_parfors_unsupported
    def test_blackscholes(self):
        # blackscholes takes 5 1D float array args
        args = (numba.float64[:], ) * 5
        self.assertTrue(countParfors(blackscholes_impl, args) == 1)

    @skip_parfors_unsupported
    @needs_blas
    def test_logistic_regression(self):
        args = (numba.float64[:], numba.float64[:,:], numba.float64[:],
                numba.int64)
        self.assertTrue(countParfors(lr_impl, args) == 1)
        self.assertTrue(countArrayAllocs(lr_impl, args) == 1)

    @skip_parfors_unsupported
    def test_kmeans(self):
        np.random.seed(0)
        N = 1024
        D = 10
        centers = 3
        A = np.random.ranf((N, D))
        init_centroids = np.random.ranf((centers, D))
        self.check(example_kmeans_test, A, centers, 3, init_centroids,
                                                                    decimal=1)
        # TODO: count parfors after k-means fusion is working
        # requires recursive parfor counting
        arg_typs = (types.Array(types.float64, 2, 'C'), types.intp, types.intp,
                    types.Array(types.float64, 2, 'C'))
        self.assertTrue(
            countNonParforArrayAccesses(example_kmeans_test, arg_typs) == 0)

    @unittest.skipIf(not _32bit, "Only impacts 32 bit hardware")
    @needs_blas
    def test_unsupported_combination_raises(self):
        """
        This test is in place until issues with the 'parallel'
        target on 32 bit hardware are fixed.
        """
        with self.assertRaises(errors.UnsupportedParforsError) as raised:
            @njit(parallel=True)
            def ddot(a, v):
                return np.dot(a, v)

            A = np.linspace(0, 1, 20).reshape(2, 10)
            v = np.linspace(2, 1, 10)
            ddot(A, v)

        msg = ("The 'parallel' target is not currently supported on 32 bit "
               "hardware")
        self.assertIn(msg, str(raised.exception))

    @skip_parfors_unsupported
    def test_simple01(self):
        def test_impl():
            return np.ones(())
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl)
        self.assertIn("\'@do_scheduling\' not found", str(raises.exception))

    @skip_parfors_unsupported
    def test_simple02(self):
        def test_impl():
            return np.ones((1,))
        self.check(test_impl)

    @skip_parfors_unsupported
    def test_simple03(self):
        def test_impl():
            return np.ones((1, 2))
        self.check(test_impl)

    @skip_parfors_unsupported
    def test_simple04(self):
        def test_impl():
            return np.ones(1)
        self.check(test_impl)

    @skip_parfors_unsupported
    def test_simple07(self):
        def test_impl():
            return np.ones((1, 2), dtype=np.complex128)
        self.check(test_impl)

    @skip_parfors_unsupported
    def test_simple08(self):
        def test_impl():
            return np.ones((1, 2)) + np.ones((1, 2))
        self.check(test_impl)

    @skip_parfors_unsupported
    def test_simple09(self):
        def test_impl():
            return np.ones((1, 1))
        self.check(test_impl)

    @skip_parfors_unsupported
    def test_simple10(self):
        def test_impl():
            return np.ones((0, 0))
        self.check(test_impl)

    @skip_parfors_unsupported
    def test_simple11(self):
        def test_impl():
            return np.ones((10, 10)) + 1.
        self.check(test_impl)

    @skip_parfors_unsupported
    def test_simple12(self):
        def test_impl():
            return np.ones((10, 10)) + np.complex128(1.)
        self.check(test_impl)

    @skip_parfors_unsupported
    def test_simple13(self):
        def test_impl():
            return np.complex128(1.)
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl)
        self.assertIn("\'@do_scheduling\' not found", str(raises.exception))

    @skip_parfors_unsupported
    def test_simple14(self):
        def test_impl():
            return np.ones((10, 10))[0::20]
        self.check(test_impl)

    @skip_parfors_unsupported
    def test_simple15(self):
        def test_impl(v1, v2, m1, m2):
            return v1 + v1
        self.check(test_impl, *self.simple_args)

    @skip_parfors_unsupported
    def test_simple16(self):
        def test_impl(v1, v2, m1, m2):
            return m1 + m1
        self.check(test_impl, *self.simple_args)

    @skip_parfors_unsupported
    def test_simple17(self):
        def test_impl(v1, v2, m1, m2):
            return m2 + v1
        self.check(test_impl, *self.simple_args)

    @skip_parfors_unsupported
    @needs_lapack
    def test_simple18(self):
        def test_impl(v1, v2, m1, m2):
            return m1.T + np.linalg.svd(m2)[1]
        self.check(test_impl, *self.simple_args)

    @skip_parfors_unsupported
    @needs_blas
    def test_simple19(self):
        def test_impl(v1, v2, m1, m2):
            return np.dot(m1, v2)
        self.check(test_impl, *self.simple_args)

    @skip_parfors_unsupported
    @needs_blas
    def test_simple20(self):
        def test_impl(v1, v2, m1, m2):
            return np.dot(m1, m2)
        # gemm is left to BLAS
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl, *self.simple_args)
        self.assertIn("\'@do_scheduling\' not found", str(raises.exception))

    @skip_parfors_unsupported
    @needs_blas
    def test_simple21(self):
        def test_impl(v1, v2, m1, m2):
            return np.dot(v1, v1)
        self.check(test_impl, *self.simple_args)

    @skip_parfors_unsupported
    def test_simple22(self):
        def test_impl(v1, v2, m1, m2):
            return np.sum(v1 + v1)
        self.check(test_impl, *self.simple_args)

    @skip_parfors_unsupported
    def test_simple23(self):
        def test_impl(v1, v2, m1, m2):
            x = 2 * v1
            y = 2 * v1
            return 4 * np.sum(x**2 + y**2 < 1) / 10
        self.check(test_impl, *self.simple_args)

    @skip_parfors_unsupported
    def test_simple24(self):
        def test_impl():
            n = 20
            A = np.ones((n, n))
            b = np.arange(n)
            return np.sum(A[:, b])
        self.check(test_impl)

    @disabled_test
    def test_simple_operator_15(self):
        """same as corresponding test_simple_<n> case but using operator.add"""
        def test_impl(v1, v2, m1, m2):
            return operator.add(v1, v1)

        self.check(test_impl, *self.simple_args)

    @disabled_test
    def test_simple_operator_16(self):
        def test_impl(v1, v2, m1, m2):
            return operator.add(m1, m1)

        self.check(test_impl, *self.simple_args)

    @disabled_test
    def test_simple_operator_17(self):
        def test_impl(v1, v2, m1, m2):
            return operator.add(m2, v1)

        self.check(test_impl, *self.simple_args)

    @skip_parfors_unsupported
    def test_np_func_direct_import(self):
        from numpy import ones  # import here becomes FreeVar
        def test_impl(n):
            A = ones(n)
            return A[0]
        n = 111
        self.check(test_impl, n)

    @skip_parfors_unsupported
    def test_np_random_func_direct_import(self):
        def test_impl(n):
            A = randn(n)
            return A[0]
        self.assertTrue(countParfors(test_impl, (types.int64, )) == 1)

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
    def test_var(self):
        def test_impl(A):
            return A.var()
        N = 100
        A = np.random.ranf(N)
        B = np.random.randint(10, size=(N, 3))
        C = A + 1j * A
        self.check(test_impl, A)
        self.check(test_impl, B)
        self.check(test_impl, C)
        self.assertTrue(countParfors(test_impl, (types.Array(types.float64, 1, 'C'), )) == 2)
        self.assertTrue(countParfors(test_impl, (types.Array(types.float64, 2, 'C'), )) == 2)

    @skip_parfors_unsupported
    def test_std(self):
        def test_impl(A):
            return A.std()
        N = 100
        A = np.random.ranf(N)
        B = np.random.randint(10, size=(N, 3))
        C = A + 1j * A
        self.check(test_impl, A)
        self.check(test_impl, B)
        self.check(test_impl, C)
        self.assertTrue(countParfors(test_impl, (types.Array(types.float64, 1, 'C'), )) == 2)
        self.assertTrue(countParfors(test_impl, (types.Array(types.float64, 2, 'C'), )) == 2)

    @skip_parfors_unsupported
    def test_issue4963_globals(self):
        def test_impl():
            buf = np.zeros((_GLOBAL_INT_FOR_TESTING1, _GLOBAL_INT_FOR_TESTING2))
            return buf
        self.check(test_impl)

    @skip_parfors_unsupported
    def test_issue4963_freevars(self):
        _FREEVAR_INT_FOR_TESTING1 = 17
        _FREEVAR_INT_FOR_TESTING2 = 5
        def test_impl():
            buf = np.zeros((_FREEVAR_INT_FOR_TESTING1, _FREEVAR_INT_FOR_TESTING2))
            return buf
        self.check(test_impl)

    @skip_parfors_unsupported
    def test_random_parfor(self):
        """
        Test function with only a random call to make sure a random function
        like ranf is actually translated to a parfor.
        """
        def test_impl(n):
            A = np.random.ranf((n, n))
            return A
        self.assertTrue(countParfors(test_impl, (types.int64, )) == 1)

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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

        # checks that 0d array input raises
        msg = ("zero-size array to reduction operation "
               "minimum which has no identity")
        for impl in (test_impl1, test_impl2):
            pcfunc = self.compile_parallel(impl, (types.int64[:],))
            with self.assertRaises(ValueError) as e:
                pcfunc.entry_point(np.array([], dtype=np.int64))
            self.assertIn(msg, str(e.exception))

    @skip_parfors_unsupported
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

        # checks that 0d array input raises
        msg = ("zero-size array to reduction operation "
               "maximum which has no identity")
        for impl in (test_impl1, test_impl2):
            pcfunc = self.compile_parallel(impl, (types.int64[:],))
            with self.assertRaises(ValueError) as e:
                pcfunc.entry_point(np.array([], dtype=np.int64))
            self.assertIn(msg, str(e.exception))

    @skip_parfors_unsupported
    def test_use_of_reduction_var1(self):
        def test_impl():
            acc = 0
            for i in prange(1):
                acc = cmath.sqrt(acc)
            return acc

        # checks that invalid use of reduction variable is detected
        msg = ("Use of reduction variable acc in an unsupported reduction function.")
        with self.assertRaises(ValueError) as e:
            pcfunc = self.compile_parallel(test_impl, ())
        self.assertIn(msg, str(e.exception))

    @skip_parfors_unsupported
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

        # checks that 0d array input raises
        msg = 'attempt to get argmin of an empty sequence'
        for impl in (test_impl1, test_impl2):
            pcfunc = self.compile_parallel(impl, (types.int64[:],))
            with self.assertRaises(ValueError) as e:
                pcfunc.entry_point(np.array([], dtype=np.int64))
            self.assertIn(msg, str(e.exception))

    @skip_parfors_unsupported
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

        # checks that 0d array input raises
        msg = 'attempt to get argmax of an empty sequence'
        for impl in (test_impl1, test_impl2):
            pcfunc = self.compile_parallel(impl, (types.int64[:],))
            with self.assertRaises(ValueError) as e:
                pcfunc.entry_point(np.array([], dtype=np.int64))
            self.assertIn(msg, str(e.exception))

    @skip_parfors_unsupported
    def test_parfor_array_access1(self):
        # signed index of the prange generated by sum() should be replaced
        # resulting in array A to be eliminated (see issue #2846)
        def test_impl(n):
            A = np.ones(n)
            return A.sum()

        n = 211
        self.check(test_impl, n)
        self.assertEqual(countArrays(test_impl, (types.intp,)), 0)

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
    def test_parfor_array_access3(self):
        def test_impl(n):
            A = np.ones(n, np.int64)
            m = 0
            for i in numba.prange(len(A)):
                m += A[i]
                if m==2:
                    i = m

        n = 211
        with self.assertRaises(errors.UnsupportedRewriteError) as raises:
            self.check(test_impl, n)
        self.assertIn("Overwrite of parallel loop index", str(raises.exception))

    @skip_parfors_unsupported
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
            if isinstance(stmt, numba.parfors.parfor.Parfor):
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

    @skip_parfors_unsupported
    def test_parfor_dtype_type(self):
        # test array type replacement creates proper type
        def test_impl(a):
            for i in numba.prange(len(a)):
                a[i] = a.dtype.type(0)
            return a[4]

        a = np.ones(10)
        self.check(test_impl, a)

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
    @disabled_test # Test itself is problematic, see #3155
    def test_parfor_hoist_setitem(self):
        # Make sure that read of out is not hoisted.
        def test_impl(out):
            for i in prange(10):
                out[0] = 2 * out[0]
            return out[0]

        out = np.ones(1)
        self.check(test_impl, out)

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
    def test_ufunc_expr(self):
        # issue #2885
        def test_impl(A, B):
            return np.bitwise_and(A, B)

        A = np.ones(3, np.uint8)
        B = np.ones(3, np.uint8)
        B[1] = 0
        self.check(test_impl, A, B)

    @skip_parfors_unsupported
    def test_find_callname_intrinsic(self):
        def test_impl(n):
            A = unsafe_empty((n,))
            for i in range(n):
                A[i] = i + 2.0
            return A

        # the unsafe allocation should be found even though it is imported
        # as a different name
        self.assertEqual(countArrayAllocs(test_impl, (types.intp,)), 1)

    @skip_parfors_unsupported
    def test_reduction_var_reuse(self):
        # issue #3139
        def test_impl(n):
            acc = 0
            for i in prange(n):
                acc += 1

            for i in prange(n):
                acc += 2

            return acc
        self.check(test_impl, 16)

    @skip_parfors_unsupported
    def test_two_d_array_reduction_reuse(self):
        def test_impl(n):
            shp = (13, 17)
            size = shp[0] * shp[1]
            result1 = np.zeros(shp, np.int_)
            tmp = np.arange(size).reshape(shp)

            for i in numba.prange(n):
                result1 += tmp

            for i in numba.prange(n):
                result1 += tmp

            return result1

        self.check(test_impl, 100)

    @skip_parfors_unsupported
    def test_one_d_array_reduction(self):
        def test_impl(n):
            result = np.zeros(1, np.int_)

            for i in numba.prange(n):
                result += np.array([i], np.int_)

            return result

        self.check(test_impl, 100)

    @skip_parfors_unsupported
    def test_two_d_array_reduction(self):
        def test_impl(n):
            shp = (13, 17)
            size = shp[0] * shp[1]
            result1 = np.zeros(shp, np.int_)
            tmp = np.arange(size).reshape(shp)

            for i in numba.prange(n):
                result1 += tmp

            return result1

        self.check(test_impl, 100)

    @skip_parfors_unsupported
    def test_two_d_array_reduction_with_float_sizes(self):
        # result1 is float32 and tmp is float64.
        # Tests reduction with differing dtypes.
        def test_impl(n):
            shp = (2, 3)
            result1 = np.zeros(shp, np.float32)
            tmp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(shp)

            for i in numba.prange(n):
                result1 += tmp

            return result1

        self.check(test_impl, 100)

    @skip_parfors_unsupported
    def test_two_d_array_reduction_prod(self):
        def test_impl(n):
            shp = (13, 17)
            result1 = 2 * np.ones(shp, np.int_)
            tmp = 2 * np.ones_like(result1)

            for i in numba.prange(n):
                result1 *= tmp

            return result1

        self.check(test_impl, 100)

    @skip_parfors_unsupported
    def test_three_d_array_reduction(self):
        def test_impl(n):
            shp = (3, 2, 7)
            result1 = np.zeros(shp, np.int_)

            for i in numba.prange(n):
                result1 += np.ones(shp, np.int_)

            return result1

        self.check(test_impl, 100)

    @skip_parfors_unsupported
    def test_preparfor_canonicalize_kws(self):
        # test canonicalize_array_math typing for calls with kw args
        def test_impl(A):
            return A.argsort() + 1

        n = 211
        A = np.arange(n)
        self.check(test_impl, A)

    @skip_parfors_unsupported
    def test_preparfor_datetime64(self):
        # test array.dtype transformation for datetime64
        def test_impl(A):
            return A.dtype

        A = np.empty(1, np.dtype('datetime64[ns]'))
        cpfunc = self.compile_parallel(test_impl, (numba.typeof(A),))
        self.assertEqual(cpfunc.entry_point(A), test_impl(A))

    @skip_parfors_unsupported
    def test_no_hoisting_with_member_function_call(self):
        def test_impl(X):
            n = X.shape[0]
            acc = 0
            for i in prange(n):
                R = {1, 2, 3}
                R.add(i)
                tmp = 0
                for x in R:
                    tmp += x
                acc += tmp
            return acc

        self.check(test_impl, np.random.ranf(128))

    @skip_parfors_unsupported
    def test_array_compare_scalar(self):
        """ issue3671: X != 0 becomes an arrayexpr with operator.ne.
            That is turned into a parfor by devectorizing.  Make sure
            the return type of the devectorized operator.ne
            on integer types works properly.
        """
        def test_impl():
            X = np.zeros(10, dtype=np.int_)
            return X != 0

        self.check(test_impl)

    @skip_parfors_unsupported
    def test_reshape_with_neg_one(self):
        # issue3314
        def test_impl(a, b):
            result_matrix = np.zeros((b, b, 1), dtype=np.float64)
            sub_a = a[0:b]
            a = sub_a.size
            b = a / 1
            z = sub_a.reshape(-1, 1)
            result_data = sub_a / z
            result_matrix[:,:,0] = result_data
            return result_matrix

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                   7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        b = 3

        self.check(test_impl, a, b)

    @skip_parfors_unsupported
    def test_reshape_with_large_neg(self):
        # issue3314
        def test_impl(a, b):
            result_matrix = np.zeros((b, b, 1), dtype=np.float64)
            sub_a = a[0:b]
            a = sub_a.size
            b = a / 1
            z = sub_a.reshape(-1307, 1)
            result_data = sub_a / z
            result_matrix[:,:,0] = result_data
            return result_matrix

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                   7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        b = 3

        self.check(test_impl, a, b)

    @skip_parfors_unsupported
    def test_reshape_with_too_many_neg_one(self):
        # issue3314
        with self.assertRaises(errors.UnsupportedRewriteError) as raised:
            @njit(parallel=True)
            def test_impl(a, b):
                rm = np.zeros((b, b, 1), dtype=np.float64)
                sub_a = a[0:b]
                a = sub_a.size
                b = a / 1
                z = sub_a.reshape(-1, -1)
                result_data = sub_a / z
                rm[:,:,0] = result_data
                return rm

            a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                       7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            b = 3
            test_impl(a, b)

        msg = ("The reshape API may only include one negative argument.")
        self.assertIn(msg, str(raised.exception))

    @skip_parfors_unsupported
    def test_ndarray_fill(self):
        def test_impl(x):
            x.fill(7.0)
            return x
        x = np.zeros(10)
        self.check(test_impl, x)
        self.assertTrue(countParfors(test_impl, (types.Array(types.float64, 1, 'C'),)) == 1)

    @skip_parfors_unsupported
    def test_ndarray_fill2d(self):
        def test_impl(x):
            x.fill(7.0)
            return x
        x = np.zeros((2,2))
        self.check(test_impl, x)
        self.assertTrue(countParfors(test_impl, (types.Array(types.float64, 2, 'C'),)) == 1)

    @skip_parfors_unsupported
    def test_0d_array(self):
        def test_impl(n):
            return np.sum(n) + np.prod(n) + np.min(n) + np.max(n) + np.var(n)
        self.check(test_impl, np.array(7), check_scheduling=False)

    @skip_parfors_unsupported
    def test_array_analysis_optional_def(self):
        def test_impl(x, half):
            size = len(x)
            parr = x[0:size]

            if half:
                parr = x[0:size//2]

            return parr.sum()
        x = np.ones(20)
        self.check(test_impl, x, True, check_scheduling=False)

    @skip_parfors_unsupported
    def test_prange_side_effects(self):
        def test_impl(a, b):
            data = np.empty(len(a), dtype=np.float64)
            size = len(data)
            for i in numba.prange(size):
                data[i] = a[i]
            for i in numba.prange(size):
                data[i] = data[i] + b[i]
            return data

        x = np.arange(10 ** 2, dtype=float)
        y = np.arange(10 ** 2, dtype=float)

        self.check(test_impl, x, y)
        self.assertTrue(countParfors(test_impl,
                                    (types.Array(types.float64, 1, 'C'),
                                     types.Array(types.float64, 1, 'C'))) == 1)

    @skip_parfors_unsupported
    def test_tuple1(self):
        def test_impl(a):
            atup = (3, 4)
            b = 7
            for i in numba.prange(len(a)):
                a[i] += atup[0] + atup[1] + b
            return a

        x = np.arange(10)
        self.check(test_impl, x)

    @skip_parfors_unsupported
    def test_tuple2(self):
        def test_impl(a):
            atup = a.shape
            b = 7
            for i in numba.prange(len(a)):
                a[i] += atup[0] + b
            return a

        x = np.arange(10)
        self.check(test_impl, x)

    @skip_parfors_unsupported
    def test_tuple3(self):
        def test_impl(a):
            atup = (np.arange(10), 4)
            b = 7
            for i in numba.prange(len(a)):
                a[i] += atup[0][5] + atup[1] + b
            return a

        x = np.arange(10)
        self.check(test_impl, x)

    @skip_parfors_unsupported
    def test_namedtuple1(self):
        def test_impl(a):
            antup = TestNamedTuple(part0=3, part1=4)
            b = 7
            for i in numba.prange(len(a)):
                a[i] += antup.part0 + antup.part1 + b
            return a

        x = np.arange(10)
        self.check(test_impl, x)

    @skip_parfors_unsupported
    def test_namedtuple2(self):
        TestNamedTuple2 = namedtuple('TestNamedTuple2', ('part0', 'part1'))
        def test_impl(a):
            antup = TestNamedTuple2(part0=3, part1=4)
            b = 7
            for i in numba.prange(len(a)):
                a[i] += antup.part0 + antup.part1 + b
            return a

        x = np.arange(10)
        self.check(test_impl, x)


class TestParforsLeaks(MemoryLeakMixin, TestParforsBase):
    def check(self, pyfunc, *args, **kwargs):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    @skip_parfors_unsupported
    def test_reduction(self):
        # issue4299
        @njit(parallel=True)
        def test_impl(arr):
            return arr.sum()

        arr = np.arange(10).astype(np.float64)
        self.check(test_impl, arr)

    @skip_parfors_unsupported
    def test_multiple_reduction_vars(self):
        @njit(parallel=True)
        def test_impl(arr):
            a = 0.
            b = 1.
            for i in prange(arr.size):
                a += arr[i]
                b += 1. / (arr[i] + 1)
            return a * b
        arr = np.arange(10).astype(np.float64)
        self.check(test_impl, arr)


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
            for instr in dis.Bytecode(pyfunc_code):
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

        if utils.PYVERSION >= (3, 8):
            co_args.append(pyfunc_code.co_posonlyargcount)
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
        with warnings.catch_warnings(record=True) as raised_warnings:
            warnings.simplefilter('always')
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
        return raised_warnings


class TestPrange(TestPrangeBase):
    """ Tests Prange """

    @skip_parfors_unsupported
    def test_prange01(self):
        def test_impl():
            n = 4
            A = np.zeros(n)
            for i in range(n):
                A[i] = 2.0 * i
            return A
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_parfors_unsupported
    def test_prange02(self):
        def test_impl():
            n = 4
            A = np.zeros(n - 1)
            for i in range(1, n):
                A[i - 1] = 2.0 * i
            return A
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_parfors_unsupported
    def test_prange03(self):
        def test_impl():
            s = 10
            for i in range(10):
                s += 2
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_parfors_unsupported
    def test_prange03mul(self):
        def test_impl():
            s = 3
            for i in range(10):
                s *= 2
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_parfors_unsupported
    def test_prange03sub(self):
        def test_impl():
            s = 100
            for i in range(10):
                s -= 2
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_parfors_unsupported
    def test_prange03div(self):
        def test_impl():
            s = 10
            for i in range(10):
                s /= 2
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
    @unittest.skip("list append is not thread-safe yet (#2391, #2408)")
    def test_prange11(self):
        def test_impl():
            n = 4
            return [np.sin(j) for j in range(n)]
        self.prange_tester(test_impl, scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
    def test_prange13(self):
        def test_impl(n):
            acc = 0
            for i in range(n):
                acc += 1
            return acc
        self.prange_tester(test_impl, np.int32(4), scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
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
    @skip_parfors_unsupported
    def test_prange16(self):
        def test_impl(N):
            acc = 0
            for i in range(-N, N):
                acc += 2
            return acc
        self.prange_tester(test_impl, 1024, scheduler_type='signed',
                           check_fastmath=True)

    @skip_parfors_unsupported
    def test_prange17(self):
        def test_impl(N):
            acc = 0
            X = np.ones(N)
            for i in range(-N, N):
                acc += X[i]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed',
                           check_fastmath=True)

    @skip_parfors_unsupported
    def test_prange18(self):
        def test_impl(N):
            acc = 0
            X = np.ones(N)
            for i in range(-N, 5):
                acc += X[i]
                for j in range(-4, N):
                    acc += X[j]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed',
                           check_fastmath=True)

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
    def test_prange20(self):
        def test_impl(N):
            acc = 0
            X = np.ones(N)
            for i in range(-1, N):
                acc += X[i]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed',
                           check_fastmath=True)

    @skip_parfors_unsupported
    def test_prange21(self):
        def test_impl(N):
            acc = 0
            for i in range(-3, -1):
                acc += 3
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed',
                           check_fastmath=True)

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
    def test_prange23(self):
        # test non-contig input
        def test_impl(A):
            for i in range(len(A)):
                A[i] = i
            return A
        A = np.zeros(32)[::2]
        self.prange_tester(test_impl, A, scheduler_type='unsigned',
                           check_fastmath=True, check_fastmath_result=True)

    @skip_parfors_unsupported
    def test_prange24(self):
        # test non-contig input, signed range
        def test_impl(A):
            for i in range(-len(A), 0):
                A[i] = i
            return A
        A = np.zeros(32)[::2]
        self.prange_tester(test_impl, A, scheduler_type='signed',
                           check_fastmath=True, check_fastmath_result=True)

    @skip_parfors_unsupported
    def test_prange25(self):
        def test_impl(A):
            n = len(A)
            buf = [np.zeros_like(A) for _ in range(n)]
            for i in range(n):
                buf[i] = A + i
            return buf
        A = np.ones((10,))
        self.prange_tester(test_impl, A,  patch_instance=[1],
                           scheduler_type='unsigned', check_fastmath=True,
                           check_fastmath_result=True)

        cpfunc = self.compile_parallel(test_impl, (numba.typeof(A),))
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        hoisted_allocs = diagnostics.hoisted_allocations()
        self.assertEqual(len(hoisted_allocs), 0)

    # should this work?
    @skip_parfors_unsupported
    def test_prange26(self):
        def test_impl(A):
            B = A[::3]
            for i in range(len(B)):
                B[i] = i
            return A
        A = np.zeros(32)[::2]
        self.prange_tester(test_impl, A, scheduler_type='unsigned',
                           check_fastmath=True, check_fastmath_result=True)

    @skip_parfors_unsupported
    def test_prange27(self):
        # issue5597: usedef error in parfor
        def test_impl(a, b, c):
            for j in range(b[0]-1):
                for k in range(2):
                    z = np.abs(a[c-1:c+1])
            return 0

        # patch inner loop to 'prange'
        self.prange_tester(test_impl,
                           np.arange(20),
                           np.asarray([4,4,4,4,4,4,4,4,4,4]),
                           0,
                           patch_instance=[1],
                           scheduler_type='unsigned',
                           check_fastmath=True)

    @skip_parfors_unsupported
    def test_prange_two_instances_same_reduction_var(self):
        # issue4922 - multiple uses of same reduction variable
        def test_impl(n):
            c = 0
            for i in range(n):
                c += 1
                if i > 10:
                    c += 1
            return c
        self.prange_tester(test_impl, 9)

    @skip_parfors_unsupported
    def test_prange_conflicting_reduction_ops(self):
        def test_impl(n):
            c = 0
            for i in range(n):
                c += 1
                if i > 10:
                    c *= 1
            return c

        with self.assertRaises(errors.UnsupportedError) as raises:
            self.prange_tester(test_impl, 9)
        msg = ('Reduction variable c has multiple conflicting reduction '
               'operators.')
        self.assertIn(msg, str(raises.exception))

#    @skip_parfors_unsupported
    @disabled_test
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


    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
    def test_prange_raises_invalid_step_size(self):
        def test_impl(N):
            acc = 0
            for i in range(0, N, 2):
                acc += 2
            return acc

        with self.assertRaises(errors.UnsupportedRewriteError) as raises:
            self.prange_tester(test_impl, 1024)
        msg = 'Only constant step size of 1 is supported for prange'
        self.assertIn(msg, str(raises.exception))

    @skip_parfors_unsupported
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
        _id = '%[A-Z_0-9]?(.[0-9]+)+[.]?[i]?'
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

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
    def test_parfor_alias1(self):
        def test_impl(n):
            b = np.zeros((n, n))
            a = b[0]
            for j in range(n):
                a[j] = j + 1
            return b.sum()
        self.prange_tester(test_impl, 4)

    @skip_parfors_unsupported
    def test_parfor_alias2(self):
        def test_impl(n):
            b = np.zeros((n, n))
            for i in range(n):
              a = b[i]
              for j in range(n):
                a[j] = i + j
            return b.sum()
        self.prange_tester(test_impl, 4)

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
    def test_parfor_race_1(self):
        def test_impl(x, y):
            for j in range(y):
                k = x
            return k
        raised_warnings = self.prange_tester(test_impl, 10, 20)
        warning_obj = raised_warnings[0]
        expected_msg = ("Variable k used in parallel loop may be written to "
                        "simultaneously by multiple workers and may result "
                        "in non-deterministic or unintended results.")
        self.assertIn(expected_msg, str(warning_obj.message))

    @skip_parfors_unsupported
    def test_nested_parfor_push_call_vars(self):
        """ issue 3686: if a prange has something inside it that causes
            a nested parfor to be generated and both the inner and outer
            parfor use the same call variable defined outside the parfors
            then ensure that when that call variable is pushed into the
            parfor that the call variable isn't duplicated with the same
            name resulting in a redundant type lock.
        """
        def test_impl():
            B = 0
            f = np.negative
            for i in range(1):
                this_matters = f(1.)
                B += f(np.zeros(1,))[0]
            for i in range(2):
                this_matters = f(1.)
                B += f(np.zeros(1,))[0]

            return B
        self.prange_tester(test_impl)

    @skip_parfors_unsupported
    def test_copy_global_for_parfor(self):
        """ issue4903: a global is copied next to a parfor so that
            it can be inlined into the parfor and thus not have to be
            passed to the parfor (i.e., an unsupported function type).
            This global needs to be renamed in the block into which
            it is copied.
        """
        def test_impl(zz, tc):
            lh = np.zeros(len(tc))
            lc = np.zeros(len(tc))
            for i in range(1):
                nt = tc[i]
                for t in range(nt):
                    lh += np.exp(zz[i, t])
                for t in range(nt):
                    lc += np.exp(zz[i, t])
            return lh, lc

        m = 2
        zz = np.ones((m, m, m))
        tc = np.ones(m, dtype=np.int_)
        self.prange_tester(test_impl, zz, tc, patch_instance=[0])

    @skip_parfors_unsupported
    def test_multiple_call_getattr_object(self):
        def test_impl(n):
            B = 0
            f = np.negative
            for i in range(1):
                this_matters = f(1.0)
                B += f(n)

            return B
        self.prange_tester(test_impl, 1.0)

    @skip_parfors_unsupported
    def test_argument_alias_recarray_field(self):
        # Test for issue4007.
        def test_impl(n):
            for i in range(len(n)):
                n.x[i] = 7.0
            return n
        X1 = np.zeros(10, dtype=[('x', float), ('y', int), ])
        X2 = np.zeros(10, dtype=[('x', float), ('y', int), ])
        X3 = np.zeros(10, dtype=[('x', float), ('y', int), ])
        v1 = X1.view(np.recarray)
        v2 = X2.view(np.recarray)
        v3 = X3.view(np.recarray)

        # Numpy doesn't seem to support almost equal on recarray.
        # So, we convert to list and use assertEqual instead.
        python_res = list(test_impl(v1))
        njit_res = list(njit(test_impl)(v2))
        pa_func = njit(test_impl, parallel=True)
        pa_res = list(pa_func(v3))
        self.assertEqual(python_res, njit_res)
        self.assertEqual(python_res, pa_res)

    @skip_parfors_unsupported
    def test_mutable_list_param(self):
        """ issue3699: test that mutable variable to call in loop
            is not hoisted.  The call in test_impl forces a manual
            check here rather than using prange_tester.
        """
        @njit
        def list_check(X):
            """ If the variable X is hoisted in the test_impl prange
                then subsequent list_check calls would return increasing
                values.
            """
            ret = X[-1]
            a = X[-1] + 1
            X.append(a)
            return ret
        def test_impl(n):
            for i in prange(n):
                X = [100]
                a = list_check(X)
            return a
        python_res = test_impl(10)
        njit_res = njit(test_impl)(10)
        pa_func = njit(test_impl, parallel=True)
        pa_res = pa_func(10)
        self.assertEqual(python_res, njit_res)
        self.assertEqual(python_res, pa_res)

    @skip_parfors_unsupported
    def test_list_comprehension_prange(self):
        # issue4569
        def test_impl(x):
            return np.array([len(x[i]) for i in range(len(x))])
        x = [np.array([1,2,3], dtype=int),np.array([1,2], dtype=int)]
        self.prange_tester(test_impl, x)

    @skip_parfors_unsupported
    def test_ssa_false_reduction(self):
        # issue5698
        # SSA for h creates assignments to h that make it look like a
        # reduction variable except that it lacks an associated
        # reduction operator.  Test here that h is excluded as a
        # reduction variable.
        def test_impl(image, a, b):
            empty = np.zeros(image.shape)
            for i in range(image.shape[0]):
                r = image[i][0] / 255.0
                if a == 0:
                    h = 0
                if b == 0:
                    h = 0
                empty[i] = [h, h, h]
            return empty

        image = np.zeros((3, 3), dtype=np.int32)
        self.prange_tester(test_impl, image, 0, 0)


@skip_parfors_unsupported
@x86_only
class TestParforsVectorizer(TestPrangeBase):

    # env mutating test
    _numba_parallel_test_ = False

    def get_gufunc_asm(self, func, schedule_type, *args, **kwargs):

        fastmath = kwargs.pop('fastmath', False)
        cpu_name = kwargs.pop('cpu_name', 'skylake-avx512')
        assertions = kwargs.pop('assertions', True)
        # force LLVM to use zmm registers for vectorization
        # https://reviews.llvm.org/D67259
        cpu_features = kwargs.pop('cpu_features', '-prefer-256-bit')

        env_opts = {'NUMBA_CPU_NAME': cpu_name,
                    'NUMBA_CPU_FEATURES': cpu_features,
                    }

        overrides = []
        for k, v in env_opts.items():
            overrides.append(override_env_config(k, v))

        with overrides[0], overrides[1]:
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
                self.assertGreaterEqual(len(matches), 1) # at least 1 parfor call
                self.assertEqual(matches[0], schedule_type)
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

        # Boundschecking breaks vectorization
        with override_env_config('NUMBA_BOUNDSCHECK', '0'):
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
    @skip_parfors_unsupported
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

        # Boundschecking breaks the diff check below because of the pickled exception
        with override_env_config('NUMBA_BOUNDSCHECK', '0'):
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

    @skip_parfors_unsupported
    def test_parfor_slice1(self):
        def test_impl(a):
            (n,) = a.shape
            b = a[0:n-2] + a[1:n-1]
            return b

        self.check(test_impl, np.ones(10))

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
    def test_parfor_slice3(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a[0:m-1,0:n-1] + a[1:m,1:n]
            return b

        self.check(test_impl, np.ones((4,3)))

    @skip_parfors_unsupported
    def test_parfor_slice4(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a[:,0:n-1] + a[:,1:n]
            return b

        self.check(test_impl, np.ones((4,3)))

    @skip_parfors_unsupported
    def test_parfor_slice5(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a[0:m-1,:] + a[1:m,:]
            return b

        self.check(test_impl, np.ones((4,3)))

    @skip_parfors_unsupported
    def test_parfor_slice6(self):
        def test_impl(a):
            b = a.transpose()
            c = a[1,:] + b[:,1]
            return c

        self.check(test_impl, np.ones((4,3)))

    @skip_parfors_unsupported
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

#    @skip_parfors_unsupported
    @disabled_test
    def test_parfor_slice8(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.transpose()
            b[1:m,1:n] = a[1:m,1:n]
            return b

        self.check(test_impl, np.arange(9).reshape((3,3)))

#    @skip_parfors_unsupported
    @disabled_test
    def test_parfor_slice9(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.transpose()
            b[1:n,1:m] = a[:,1:m]
            return b

        self.check(test_impl, np.arange(12).reshape((3,4)))

#    @skip_parfors_unsupported
    @disabled_test
    def test_parfor_slice10(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.transpose()
            b[2,1:m] = a[2,1:m]
            return b

        self.check(test_impl, np.arange(9).reshape((3,3)))

    @skip_parfors_unsupported
    def test_parfor_slice11(self):
        def test_impl(a):
            (m,n,l) = a.shape
            b = a.copy()
            b[:,1,1:l] = a[:,2,1:l]
            return b

        self.check(test_impl, np.arange(27).reshape((3,3,3)))

    @skip_parfors_unsupported
    def test_parfor_slice12(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.copy()
            b[1,1:-1] = a[0,:-2]
            return b

        self.check(test_impl, np.arange(12).reshape((3,4)))

    @skip_parfors_unsupported
    def test_parfor_slice13(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.copy()
            c = -1
            b[1,1:c] = a[0,-n:c-1]
            return b

        self.check(test_impl, np.arange(12).reshape((3,4)))

    @skip_parfors_unsupported
    def test_parfor_slice14(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.copy()
            b[1,:-1] = a[0,-3:4]
            return b

        self.check(test_impl, np.arange(12).reshape((3,4)))

    @skip_parfors_unsupported
    def test_parfor_slice15(self):
        def test_impl(a):
            (m,n) = a.shape
            b = a.copy()
            b[1,-(n-1):] = a[0,-3:4]
            return b

        self.check(test_impl, np.arange(12).reshape((3,4)))


    @disabled_test
    def test_parfor_slice16(self):
        """ This test is disabled because if n is larger than the array size
            then n and n-1 will both be the end of the array and thus the
            slices will in fact be of different sizes and unable to fuse.
        """
        def test_impl(a, b, n):
            assert(a.shape == b.shape)
            a[1:n] = 10
            b[0:(n-1)] = 10
            return a * b

        self.check(test_impl, np.ones(10), np.zeros(10), 8)
        args = (numba.float64[:], numba.float64[:], numba.int64)
        self.assertEqual(countParfors(test_impl, args), 2)

    @skip_parfors_unsupported
    def test_parfor_slice17(self):
        def test_impl(m, A):
            B = np.zeros(m)
            n = len(A)
            B[-n:] = A
            return B

        self.check(test_impl, 10, np.ones(10))

    @skip_parfors_unsupported
    def test_parfor_slice18(self):
        # issue 3534
        def test_impl():
            a = np.zeros(10)
            a[1:8] = np.arange(0, 7)
            y = a[3]
            return y

        self.check(test_impl)

    @skip_parfors_unsupported
    def test_parfor_slice19(self):
        # issues #3561 and #3554, empty slice binop
        def test_impl(X):
            X[:0] += 1
            return X

        self.check(test_impl, np.ones(10))

    @skip_parfors_unsupported
    def test_parfor_slice20(self):
        # issue #4075, slice size
        def test_impl():
            a = np.ones(10)
            c = a[1:]
            s = len(c)
            return s

        self.check(test_impl, check_scheduling=False)

    @skip_parfors_unsupported
    def test_parfor_slice21(self):
        def test_impl(x1, x2):
            x1 = x1.reshape(x1.size, 1)
            x2 = x2.reshape(x2.size, 1)
            return x1 >= x2[:-1, :]

        x1 = np.random.rand(5)
        x2 = np.random.rand(6)
        self.check(test_impl, x1, x2)

    @skip_parfors_unsupported
    def test_parfor_slice22(self):
        def test_impl(x1, x2):
            b = np.zeros((10,))
            for i in prange(1):
                b += x1[:, x2]
            return b

        x1 = np.zeros((10,7))
        x2 = np.array(4)
        self.check(test_impl, x1, x2)

    @skip_parfors_unsupported
    def test_parfor_slice23(self):
        # issue #4630
        def test_impl(x):
            x[:0] = 2
            return x

        self.check(test_impl, np.ones(10))

    @skip_parfors_unsupported
    def test_parfor_slice24(self):
        def test_impl(m, A, n):
            B = np.zeros(m)
            C = B[n:]
            C = A[:len(C)]
            return B

        for i in range(-15, 15):
            self.check(test_impl, 10, np.ones(10), i)

    @skip_parfors_unsupported
    def test_parfor_slice25(self):
        def test_impl(m, A, n):
            B = np.zeros(m)
            C = B[:n]
            C = A[:len(C)]
            return B

        for i in range(-15, 15):
            self.check(test_impl, 10, np.ones(10), i)

    @skip_parfors_unsupported
    def test_parfor_slice26(self):
        def test_impl(a):
            (n,) = a.shape
            b = a.copy()
            b[-(n-1):] = a[-3:4]
            return b

        self.check(test_impl, np.arange(4))

    @skip_parfors_unsupported
    def test_parfor_slice27(self):
        # issue5601: tests array analysis of the slice with
        # n_valid_vals of unknown size.
        def test_impl(a):
            n_valid_vals = 0

            for i in prange(a.shape[0]):
                if a[i] != 0:
                    n_valid_vals += 1

                if n_valid_vals:
                    unused = a[:n_valid_vals]

            return 0

        self.check(test_impl, np.arange(3))


class TestParforsOptions(TestParforsBase):

    def check(self, pyfunc, *args, **kwargs):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    @skip_parfors_unsupported
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

    @skip_parfors_unsupported
    def test_parfor_bitmask1(self):
        def test_impl(a, n):
            b = a > n
            a[b] = 0
            return a

        self.check(test_impl, np.arange(10), 5)

    @skip_parfors_unsupported
    def test_parfor_bitmask2(self):
        def test_impl(a, b):
            a[b] = 0
            return a

        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    @skip_parfors_unsupported
    def test_parfor_bitmask3(self):
        def test_impl(a, b):
            a[b] = a[b]
            return a

        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    @skip_parfors_unsupported
    def test_parfor_bitmask4(self):
        def test_impl(a, b):
            a[b] = (2 * a)[b]
            return a

        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    @skip_parfors_unsupported
    def test_parfor_bitmask5(self):
        def test_impl(a, b):
            a[b] = a[b] * a[b]
            return a

        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    @skip_parfors_unsupported
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

class TestParforsMisc(TestParforsBase):
    """
    Tests miscellaneous parts of ParallelAccelerator use.
    """
    _numba_parallel_test_ = False

    def check(self, pyfunc, *args, **kwargs):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    @skip_parfors_unsupported
    def test_no_warn_if_cache_set(self):

        def pyfunc():
            arr = np.ones(100)
            for i in prange(arr.size):
                arr[i] += i
            return arr

        cfunc = njit(parallel=True, cache=True)(pyfunc)

        with warnings.catch_warnings(record=True) as raised_warnings:
            warnings.simplefilter('always')
            cfunc()

        self.assertEqual(len(raised_warnings), 0)

        # Make sure the dynamic globals flag is set
        has_dynamic_globals = [cres.library.has_dynamic_globals
                               for cres in cfunc.overloads.values()]
        self.assertEqual(has_dynamic_globals, [False])

    @skip_parfors_unsupported
    def test_statement_reordering_respects_aliasing(self):
        def impl():
            a = np.zeros(10)
            a[1:8] = np.arange(0, 7)
            print('a[3]:', a[3])
            print('a[3]:', a[3])
            return a

        cres = self.compile_parallel(impl, ())
        with captured_stdout() as stdout:
            cres.entry_point()
        for line in stdout.getvalue().splitlines():
            self.assertEqual('a[3]: 2.0', line)

    @skip_parfors_unsupported
    def test_parfor_ufunc_typing(self):
        def test_impl(A):
            return np.isinf(A)

        A = np.array([np.inf, 0.0])
        cfunc = njit(parallel=True)(test_impl)
        # save global state
        old_seq_flag = numba.parfors.parfor.sequential_parfor_lowering
        try:
            numba.parfors.parfor.sequential_parfor_lowering = True
            np.testing.assert_array_equal(test_impl(A), cfunc(A))
        finally:
            # recover global state
            numba.parfors.parfor.sequential_parfor_lowering = old_seq_flag

    @skip_parfors_unsupported
    def test_init_block_dce(self):
        # issue4690
        def test_impl():
            res = 0
            arr = [1,2,3,4,5]
            numba.parfors.parfor.init_prange()
            dummy = arr
            for i in numba.prange(5):
                res += arr[i]
            return res + dummy[2]

        self.assertTrue(get_init_block_size(test_impl, ()) == 0)

    @skip_parfors_unsupported
    def test_alias_analysis_for_parfor1(self):
        def test_impl():
            acc = 0
            for _ in range(4):
                acc += 1

            data = np.zeros((acc,))
            return data

        self.check(test_impl)

    @skip_parfors_unsupported
    def test_no_state_change_in_gufunc_lowering_on_error(self):
        # tests #5098, if there's an exception arising in gufunc lowering the
        # sequential_parfor_lowering global variable should remain as False on
        # stack unwind.

        @register_pass(mutates_CFG=True, analysis_only=False)
        class BreakParfors(AnalysisPass):
            _name = "break_parfors"

            def __init__(self):
                AnalysisPass.__init__(self)

            def run_pass(self, state):
                for blk in state.func_ir.blocks.values():
                    for stmt in blk.body:
                        if isinstance(stmt, numba.parfors.parfor.Parfor):
                            # races should be a set(), that list is iterable
                            # permits it to get through to the
                            # _create_gufunc_for_parfor_body routine at which
                            # point it needs to be a set so e.g. set.difference
                            # can be computed, this therefore creates an error
                            # in the right location.
                            stmt.races = []
                    return True


        class BreakParforsCompiler(CompilerBase):

            def define_pipelines(self):
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                pm.add_pass_after(BreakParfors, IRLegalization)
                pm.finalize()
                return [pm]


        @njit(parallel=True, pipeline_class=BreakParforsCompiler)
        def foo():
            x = 1
            for _ in prange(1):
                x += 1
            return x

        # assert default state for global
        self.assertFalse(numba.parfors.parfor.sequential_parfor_lowering)

        with self.assertRaises(errors.LoweringError) as raises:
            foo()

        self.assertIn("'list' object has no attribute 'difference'",
                      str(raises.exception))

        # assert state has not changed
        self.assertFalse(numba.parfors.parfor.sequential_parfor_lowering)

    @skip_parfors_unsupported
    def test_issue_5098(self):
        class DummyType(types.Opaque):
            pass

        dummy_type = DummyType("my_dummy")
        register_model(DummyType)(models.OpaqueModel)

        class Dummy(object):
            pass

        @typeof_impl.register(Dummy)
        def typeof_Dummy(val, c):
            return dummy_type

        @unbox(DummyType)
        def unbox_index(typ, obj, c):
            return NativeValue(c.context.get_dummy_value())

        @overload_method(DummyType, "method1", jit_options={"parallel":True})
        def _get_method1(obj, arr, func):
            def _foo(obj, arr, func):
                def baz(a, f):
                    c = a.copy()
                    c[np.isinf(a)] = np.nan
                    return f(c)

                length = len(arr)
                output_arr = np.empty(length, dtype=np.float64)
                for i in prange(length):
                    output_arr[i] = baz(arr[i], func)
                for i in prange(length - 1):
                    output_arr[i] += baz(arr[i], func)
                return output_arr
            return _foo

        @njit
        def bar(v):
            return v.mean()

        @njit
        def test1(d):
            return d.method1(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), bar)

        save_state = numba.parfors.parfor.sequential_parfor_lowering
        self.assertFalse(save_state)
        try:
            test1(Dummy())
            self.assertFalse(numba.parfors.parfor.sequential_parfor_lowering)
        finally:
            # always set the sequential_parfor_lowering state back to the
            # original state
            numba.parfors.parfor.sequential_parfor_lowering = save_state

    @skip_parfors_unsupported
    def test_oversized_tuple_as_arg_to_kernel(self):

        @njit(parallel=True)
        def oversize_tuple():
            big_tup = (1,2,3,4)
            z = 0
            for x in prange(10):
                z += big_tup[0]
            return z

        with override_env_config('NUMBA_PARFOR_MAX_TUPLE_SIZE', '3'):
            with self.assertRaises(errors.UnsupportedParforsError) as raises:
                oversize_tuple()

        errstr = str(raises.exception)
        self.assertIn("Use of a tuple", errstr)
        self.assertIn("in a parallel region", errstr)

    @skip_parfors_unsupported
    def test_issue5167(self):

        def ndvi_njit(img_nir, img_red):
            fillvalue = 0
            out_img = np.full(img_nir.shape, fillvalue, dtype=img_nir.dtype)
            dims = img_nir.shape
            for y in prange(dims[0]):
                for x in prange(dims[1]):
                    out_img[y, x] = ((img_nir[y, x] - img_red[y, x]) /
                                     (img_nir[y, x] + img_red[y, x]))
            return out_img

        tile_shape = (4, 4)
        array1 = np.random.uniform(low=1.0, high=10000.0, size=tile_shape)
        array2 = np.random.uniform(low=1.0, high=10000.0, size=tile_shape)
        self.check(ndvi_njit, array1, array2)

    @skip_parfors_unsupported
    def test_issue5065(self):

        def reproducer(a, dist, dist_args):
            result = np.zeros((a.shape[0], a.shape[0]), dtype=np.float32)
            for i in prange(a.shape[0]):
                for j in range(i + 1, a.shape[0]):
                    d = dist(a[i], a[j], *dist_args)
                    result[i, j] = d
                    result[j, i] = d
            return result

        @njit
        def euclidean(x, y):
            result = 0.0
            for i in range(x.shape[0]):
                result += (x[i] - y[i]) ** 2
            return np.sqrt(result)

        a = np.random.random(size=(5, 2))

        got = njit(parallel=True)(reproducer)(a.copy(), euclidean,())
        expected = reproducer(a.copy(), euclidean,())

        np.testing.assert_allclose(got, expected)

    @skip_parfors_unsupported
    def test_issue5001(self):

        def test_numba_parallel(myarray):
            result = [0] * len(myarray)
            for i in prange(len(myarray)):
                result[i] = len(myarray[i])
            return result

        myarray = (np.empty(100),np.empty(50))
        self.check(test_numba_parallel, myarray)

    @skip_parfors_unsupported
    def test_issue3169(self):

        @njit
        def foo(grids):
            pass

        @njit(parallel=True)
        def bar(grids):
            for x in prange(1):
                foo(grids)

        # returns nothing, just check it compiles
        bar(([1],) * 2)

    @disabled_test
    def test_issue4846(self):

        mytype = namedtuple("mytype", ("a", "b"))

        def outer(mydata):
            for k in prange(3):
                inner(k, mydata)
            return mydata.a

        @njit(nogil=True)
        def inner(k, mydata):
            f = (k, mydata.a)
            g = (k, mydata.b)

        mydata = mytype(a="a", b="b")

        self.check(outer, mydata)

    @skip_parfors_unsupported
    def test_issue3748(self):

        def test1b():
            x = (1, 2, 3, 4, 5)
            a = 0
            for i in prange(len(x)):
                a += x[i]
            return a

        self.check(test1b,)

    @skip_parfors_unsupported
    def test_issue5277(self):

        def parallel_test(size, arr):
            for x in prange(size[0]):
                for y in prange(size[1]):
                    arr[y][x] = x * 4.5 + y
            return arr

        size = (10, 10)
        arr = np.zeros(size, dtype=int)

        self.check(parallel_test, size, arr)

    @skip_parfors_unsupported
    def test_issue5570_ssa_races(self):
        @njit(parallel=True)
        def foo(src, method, out):
            for i in prange(1):
                for j in range(1):
                    out[i, j] = 1
            if method:
                out += 1
            return out

        src = np.zeros((5,5))
        method = 57
        out = np.zeros((2, 2))

        self.assertPreciseEqual(
            foo(src, method, out),
            foo.py_func(src, method, out)
        )


@skip_parfors_unsupported
class TestParforsDiagnostics(TestParforsBase):

    def check(self, pyfunc, *args, **kwargs):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    def assert_fusion_equivalence(self, got, expected):
        a = self._fusion_equivalent(got)
        b = self._fusion_equivalent(expected)
        self.assertEqual(a, b)

    def _fusion_equivalent(self, thing):
        # parfors indexes the Parfors class instance id's from wherever the
        # internal state happens to be. To assert fusion equivalence we just
        # check that the relative difference between fusion adjacency lists
        # is the same. For example:
        # {3: [2, 1]} is the same as {13: [12, 11]}
        # this function strips the indexing etc out returning something suitable
        # for checking equivalence
        new = defaultdict(list)
        min_key = min(thing.keys())
        for k in sorted(thing.keys()):
            new[k - min_key] = [x - min_key for x in thing[k]]
        return new

    def assert_diagnostics(self, diagnostics, parfors_count=None,
                           fusion_info=None, nested_fusion_info=None,
                           replaced_fns=None, hoisted_allocations=None):
        if parfors_count is not None:
            self.assertEqual(parfors_count, diagnostics.count_parfors())
        if fusion_info is not None:
            self.assert_fusion_equivalence(fusion_info, diagnostics.fusion_info)
        if nested_fusion_info is not None:
            self.assert_fusion_equivalence(nested_fusion_info,
                                           diagnostics.nested_fusion_info)
        if replaced_fns is not None:
            repl = diagnostics.replaced_fns.values()
            for x in replaced_fns:
                for replaced in repl:
                    if replaced[0] == x:
                        break
                else:
                    msg = "Replacement for %s was not found. Had %s" % (x, repl)
                    raise AssertionError(msg)

        if hoisted_allocations is not None:
            hoisted_allocs = diagnostics.hoisted_allocations()
            self.assertEqual(hoisted_allocations, len(hoisted_allocs))

        # just make sure that the dump() function doesn't have an issue!
        with captured_stdout():
            for x in range(1, 5):
                diagnostics.dump(x)

    def test_array_expr(self):
        def test_impl():
            n = 10
            a = np.ones(n)
            b = np.zeros(n)
            return a + b

        self.check(test_impl,)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=1,
                                fusion_info = {3: [4, 5]})

    def test_prange(self):
        def test_impl():
            n = 10
            a = np.empty(n)
            for i in prange(n):
                a[i] = i * 10
            return a

        self.check(test_impl,)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=1)

    def test_nested_prange(self):
        def test_impl():
            n = 10
            a = np.empty((n, n))
            for i in prange(n):
                for j in prange(n):
                    a[i, j] = i * 10 + j
            return a

        self.check(test_impl,)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=2,
                                nested_fusion_info={2: [1]})

    def test_function_replacement(self):
        def test_impl():
            n = 10
            a = np.ones(n)
            b = np.argmin(a)
            return b

        self.check(test_impl,)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=1,
                                fusion_info={2: [3]},
                                replaced_fns = [('argmin', 'numpy'),])

    def test_reduction(self):
        def test_impl():
            n = 10
            a = np.ones(n + 1) # prevent fusion
            acc = 0
            for i in prange(n):
                acc += a[i]
            return acc

        self.check(test_impl,)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=2)

    def test_setitem(self):
        def test_impl():
            n = 10
            a = np.ones(n)
            a[:] = 7
            return a

        self.check(test_impl,)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=1)

    def test_allocation_hoisting(self):
        def test_impl():
            n = 10
            m = 5
            acc = 0
            for i in prange(n):
                temp = np.zeros((m,)) # the np.empty call should get hoisted
                for j in range(m):
                    temp[j] = i
                acc += temp[-1]
            return acc

        self.check(test_impl,)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, hoisted_allocations=1)


if __name__ == "__main__":
    unittest.main()
