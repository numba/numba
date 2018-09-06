#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import ast
import inspect
import operator
import types as pytypes
from contextlib import contextmanager
from copy import deepcopy

import numba
from numba import unittest_support as unittest
from numba import njit, stencil, types
from numba.compiler import compile_extra, Flags
from numba.targets import registry
from numba.targets.cpu import ParallelOptions
from .support import tag
from numba.errors import LoweringError, TypingError


# for decorating tests, marking that Windows with Python 2.7 is not supported
_py27 = sys.version_info[:2] == (2, 7)
_windows_py27 = (sys.platform.startswith('win32') and _py27)
_32bit = sys.maxsize <= 2 ** 32
_reason = 'parfors not supported'
_unsupported = _32bit or _windows_py27
skip_unsupported = unittest.skipIf(_unsupported, _reason)


@stencil
def stencil1_kernel(a):
    return 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0])


@stencil(neighborhood=((-5, 0), ))
def stencil2_kernel(a):
    cum = a[-5]
    for i in range(-4, 1):
        cum += a[i]
    return 0.3 * cum


@stencil(cval=1.0)
def stencil3_kernel(a):
    return 0.25 * a[-2, 2]


@stencil
def stencil_multiple_input_kernel(a, b):
    return 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0] +
                   b[0, 1] + b[1, 0] + b[0, -1] + b[-1, 0])


@stencil
def stencil_multiple_input_kernel_var(a, b, w):
    return w * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0] +
                b[0, 1] + b[1, 0] + b[0, -1] + b[-1, 0])


@stencil(standard_indexing=("b",))
def stencil_with_standard_indexing_1d(a, b):
    return a[-1] * b[0] + a[0] * b[1]


@stencil(standard_indexing=("b",))
def stencil_with_standard_indexing_2d(a, b):
    return (a[0, 1] * b[0, 1] + a[1, 0] * b[1, 0]
            + a[0, -1] * b[0, -1] + a[-1, 0] * b[-1, 0])


@njit
def addone_njit(a):
    return a + 1

# guard against the decorator being run on unsupported platforms
# as it will raise and stop test discovery from working
if not _unsupported:
    @njit(parallel=True)
    def addone_pjit(a):
        return a + 1


class TestStencilBase(unittest.TestCase):

    _numba_parallel_test_ = False

    def __init__(self, *args):
        # flags for njit()
        self.cflags = Flags()
        self.cflags.set('nrt')

        super(TestStencilBase, self).__init__(*args)

    def _compile_this(self, func, sig, flags):
        return compile_extra(registry.cpu_target.typing_context,
                             registry.cpu_target.target_context, func, sig,
                             None, flags, {})

    def compile_parallel(self, func, sig, **kws):
        flags = Flags()
        flags.set('nrt')
        options = True if not kws else kws
        flags.set('auto_parallel', ParallelOptions(options))
        return self._compile_this(func, sig, flags)

    def compile_njit(self, func, sig):
        return self._compile_this(func, sig, flags=self.cflags)

    def compile_all(self, pyfunc, *args, **kwargs):
        sig = tuple([numba.typeof(x) for x in args])
        # compile with parallel=True
        cpfunc = self.compile_parallel(pyfunc, sig)
        # compile a standard njit of the original function
        cfunc = self.compile_njit(pyfunc, sig)
        return cfunc, cpfunc

    def check(self, no_stencil_func, pyfunc, *args):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        # results without stencil macro
        expected = no_stencil_func(*args)
        # python result
        py_output = pyfunc(*args)

        # njit result
        njit_output = cfunc.entry_point(*args)

        # parfor result
        parfor_output = cpfunc.entry_point(*args)

        np.testing.assert_almost_equal(py_output, expected, decimal=3)
        np.testing.assert_almost_equal(njit_output, expected, decimal=3)
        np.testing.assert_almost_equal(parfor_output, expected, decimal=3)

        # make sure parfor set up scheduling
        self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())


class TestStencil(TestStencilBase):

    def __init__(self, *args, **kwargs):
        super(TestStencil, self).__init__(*args, **kwargs)

    @skip_unsupported
    @tag('important')
    def test_stencil1(self):
        """Tests whether the optional out argument to stencil calls works.
        """
        def test_with_out(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.zeros(n**2).reshape((n, n))
            B = stencil1_kernel(A, out=B)
            return B

        def test_without_out(n):
            A = np.arange(n**2).reshape((n, n))
            B = stencil1_kernel(A)
            return B

        def test_impl_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.zeros(n**2).reshape((n, n))
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    B[i, j] = 0.25 * (A[i, j + 1] +
                                      A[i + 1, j] + A[i, j - 1] + A[i - 1, j])
            return B

        n = 100
        self.check(test_impl_seq, test_with_out, n)
        self.check(test_impl_seq, test_without_out, n)

    @skip_unsupported
    @tag('important')
    def test_stencil2(self):
        """Tests whether the optional neighborhood argument to the stencil
        decorate works.
        """
        def test_seq(n):
            A = np.arange(n)
            B = stencil2_kernel(A)
            return B

        def test_impl_seq(n):
            A = np.arange(n)
            B = np.zeros(n)
            for i in range(5, len(A)):
                B[i] = 0.3 * sum(A[i - 5:i + 1])
            return B

        n = 100
        self.check(test_impl_seq, test_seq, n)
        # variable length neighborhood in numba.stencil call
        # only supported in parallel path

        def test_seq(n, w):
            A = np.arange(n)

            def stencil2_kernel(a, w):
                cum = a[-w]
                for i in range(-w + 1, w + 1):
                    cum += a[i]
                return 0.3 * cum
            B = numba.stencil(stencil2_kernel, neighborhood=((-w, w), ))(A, w)
            return B

        def test_impl_seq(n, w):
            A = np.arange(n)
            B = np.zeros(n)
            for i in range(w, len(A) - w):
                B[i] = 0.3 * sum(A[i - w:i + w + 1])
            return B
        n = 100
        w = 5
        cpfunc = self.compile_parallel(test_seq, (types.intp, types.intp))
        expected = test_impl_seq(n, w)
        # parfor result
        parfor_output = cpfunc.entry_point(n, w)
        np.testing.assert_almost_equal(parfor_output, expected, decimal=3)
        self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())
        # test index_offsets

        def test_seq(n, w, offset):
            A = np.arange(n)

            def stencil2_kernel(a, w):
                cum = a[-w + 1]
                for i in range(-w + 1, w + 1):
                    cum += a[i + 1]
                return 0.3 * cum
            B = numba.stencil(stencil2_kernel, neighborhood=((-w, w), ),
                              index_offsets=(-offset, ))(A, w)
            return B

        offset = 1
        cpfunc = self.compile_parallel(test_seq, (types.intp, types.intp,
                                                  types.intp))
        parfor_output = cpfunc.entry_point(n, w, offset)
        np.testing.assert_almost_equal(parfor_output, expected, decimal=3)
        self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())
        # test slice in kernel

        def test_seq(n, w, offset):
            A = np.arange(n)

            def stencil2_kernel(a, w):
                return 0.3 * np.sum(a[-w + 1:w + 2])
            B = numba.stencil(stencil2_kernel, neighborhood=((-w, w), ),
                              index_offsets=(-offset, ))(A, w)
            return B

        offset = 1
        cpfunc = self.compile_parallel(test_seq, (types.intp, types.intp,
                                                  types.intp))
        parfor_output = cpfunc.entry_point(n, w, offset)
        np.testing.assert_almost_equal(parfor_output, expected, decimal=3)
        self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())

    @skip_unsupported
    @tag('important')
    def test_stencil3(self):
        """Tests whether a non-zero optional cval argument to the stencil
        decorator works.  Also tests integer result type.
        """
        def test_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = stencil3_kernel(A)
            return B

        test_njit = njit(test_seq)
        test_par = njit(test_seq, parallel=True)

        n = 5
        seq_res = test_seq(n)
        njit_res = test_njit(n)
        par_res = test_par(n)

        self.assertTrue(seq_res[0, 0] == 1.0 and seq_res[4, 4] == 1.0)
        self.assertTrue(njit_res[0, 0] == 1.0 and njit_res[4, 4] == 1.0)
        self.assertTrue(par_res[0, 0] == 1.0 and par_res[4, 4] == 1.0)

    @skip_unsupported
    @tag('important')
    def test_stencil_standard_indexing_1d(self):
        """Tests standard indexing with a 1d array.
        """
        def test_seq(n):
            A = np.arange(n)
            B = [3.0, 7.0]
            C = stencil_with_standard_indexing_1d(A, B)
            return C

        def test_impl_seq(n):
            A = np.arange(n)
            B = [3.0, 7.0]
            C = np.zeros(n)

            for i in range(1, n):
                C[i] = A[i - 1] * B[0] + A[i] * B[1]
            return C

        n = 100
        self.check(test_impl_seq, test_seq, n)

    @skip_unsupported
    @tag('important')
    def test_stencil_standard_indexing_2d(self):
        """Tests standard indexing with a 2d array and multiple stencil calls.
        """
        def test_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.ones((3, 3))
            C = stencil_with_standard_indexing_2d(A, B)
            D = stencil_with_standard_indexing_2d(C, B)
            return D

        def test_impl_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.ones((3, 3))
            C = np.zeros(n**2).reshape((n, n))
            D = np.zeros(n**2).reshape((n, n))

            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    C[i, j] = (A[i, j + 1] * B[0, 1] + A[i + 1, j] * B[1, 0] +
                               A[i, j - 1] * B[0, -1] + A[i - 1, j] * B[-1, 0])
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    D[i, j] = (C[i, j + 1] * B[0, 1] + C[i + 1, j] * B[1, 0] +
                               C[i, j - 1] * B[0, -1] + C[i - 1, j] * B[-1, 0])
            return D

        n = 5
        self.check(test_impl_seq, test_seq, n)

    @skip_unsupported
    @tag('important')
    def test_stencil_multiple_inputs(self):
        """Tests whether multiple inputs of the same size work.
        """
        def test_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.arange(n**2).reshape((n, n))
            C = stencil_multiple_input_kernel(A, B)
            return C

        def test_impl_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.arange(n**2).reshape((n, n))
            C = np.zeros(n**2).reshape((n, n))
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    C[i, j] = 0.25 * \
                        (A[i, j + 1] + A[i + 1, j]
                         + A[i, j - 1] + A[i - 1, j]
                         + B[i, j + 1] + B[i + 1, j]
                         + B[i, j - 1] + B[i - 1, j])
            return C

        n = 3
        self.check(test_impl_seq, test_seq, n)
        # test stencil with a non-array input

        def test_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.arange(n**2).reshape((n, n))
            w = 0.25
            C = stencil_multiple_input_kernel_var(A, B, w)
            return C
        self.check(test_impl_seq, test_seq, n)

    @skip_unsupported
    @tag('important')
    def test_stencil_call(self):
        """Tests 2D numba.stencil calls.
        """
        def test_impl1(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.zeros(n**2).reshape((n, n))
            numba.stencil(lambda a: 0.25 * (a[0, 1] + a[1, 0] + a[0, -1]
                                            + a[-1, 0]))(A, out=B)
            return B

        def test_impl2(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.zeros(n**2).reshape((n, n))

            def sf(a):
                return 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0])
            B = numba.stencil(sf)(A)
            return B

        def test_impl_seq(n):
            A = np.arange(n**2).reshape((n, n))
            B = np.zeros(n**2).reshape((n, n))
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    B[i, j] = 0.25 * (A[i, j + 1] + A[i + 1, j]
                                      + A[i, j - 1] + A[i - 1, j])
            return B

        n = 100
        self.check(test_impl_seq, test_impl1, n)
        self.check(test_impl_seq, test_impl2, n)

    @skip_unsupported
    @tag('important')
    def test_stencil_call_1D(self):
        """Tests 1D numba.stencil calls.
        """
        def test_impl(n):
            A = np.arange(n)
            B = np.zeros(n)
            numba.stencil(lambda a: 0.3 * (a[-1] + a[0] + a[1]))(A, out=B)
            return B

        def test_impl_seq(n):
            A = np.arange(n)
            B = np.zeros(n)
            for i in range(1, n - 1):
                B[i] = 0.3 * (A[i - 1] + A[i] + A[i + 1])
            return B

        n = 100
        self.check(test_impl_seq, test_impl, n)

    @skip_unsupported
    @tag('important')
    def test_stencil_call_const(self):
        """Tests numba.stencil call that has an index that can be inferred as
        constant from a unary expr. Otherwise, this would raise an error since
        neighborhood length is not specified.
        """
        def test_impl1(n):
            A = np.arange(n)
            B = np.zeros(n)
            c = 1
            numba.stencil(lambda a,c : 0.3 * (a[-c] + a[0] + a[c]))(
                                                                   A, c, out=B)
            return B

        def test_impl2(n):
            A = np.arange(n)
            B = np.zeros(n)
            c = 2
            numba.stencil(lambda a,c : 0.3 * (a[1-c] + a[0] + a[c-1]))(
                                                                   A, c, out=B)
            return B

        # recursive expr case
        def test_impl3(n):
            A = np.arange(n)
            B = np.zeros(n)
            c = 2
            numba.stencil(lambda a,c : 0.3 * (a[-c+1] + a[0] + a[c-1]))(
                                                                   A, c, out=B)
            return B

        # multi-constant case
        def test_impl4(n):
            A = np.arange(n)
            B = np.zeros(n)
            d = 1
            c = 2
            numba.stencil(lambda a,c,d : 0.3 * (a[-c+d] + a[0] + a[c-d]))(
                                                                A, c, d, out=B)
            return B

        def test_impl_seq(n):
            A = np.arange(n)
            B = np.zeros(n)
            c = 1
            for i in range(1, n - 1):
                B[i] = 0.3 * (A[i - c] + A[i] + A[i + c])
            return B

        n = 100
        # constant inference is only possible in parallel path
        cpfunc1 = self.compile_parallel(test_impl1, (types.intp,))
        cpfunc2 = self.compile_parallel(test_impl2, (types.intp,))
        cpfunc3 = self.compile_parallel(test_impl3, (types.intp,))
        cpfunc4 = self.compile_parallel(test_impl4, (types.intp,))
        expected = test_impl_seq(n)
        # parfor result
        parfor_output1 = cpfunc1.entry_point(n)
        parfor_output2 = cpfunc2.entry_point(n)
        parfor_output3 = cpfunc3.entry_point(n)
        parfor_output4 = cpfunc4.entry_point(n)
        np.testing.assert_almost_equal(parfor_output1, expected, decimal=3)
        np.testing.assert_almost_equal(parfor_output2, expected, decimal=3)
        np.testing.assert_almost_equal(parfor_output3, expected, decimal=3)
        np.testing.assert_almost_equal(parfor_output4, expected, decimal=3)

        # check error in regular Python path
        with self.assertRaises(ValueError) as e:
            test_impl4(4)

        self.assertIn("stencil kernel index is not constant, "
                      "'neighborhood' option required", str(e.exception))
        # check error in njit path
        # TODO: ValueError should be thrown instead of LoweringError
        with self.assertRaises(LoweringError) as e:
            njit(test_impl4)(4)

        self.assertIn("stencil kernel index is not constant, "
                      "'neighborhood' option required", str(e.exception))

    @skip_unsupported
    @tag('important')
    def test_stencil_parallel_off(self):
        """Tests 1D numba.stencil calls without parallel translation
           turned off.
        """
        def test_impl(A):
            return numba.stencil(lambda a: 0.3 * (a[-1] + a[0] + a[1]))(A)

        cpfunc = self.compile_parallel(test_impl, (numba.float64[:],), stencil=False)
        self.assertNotIn('@do_scheduling', cpfunc.library.get_llvm_str())



class pyStencilGenerator:
    """
    Holds the classes and methods needed to generate a python stencil
    implementation from a kernel purely using AST transforms.
    """

    class Builder:
        """
        Provides code generation for the AST manipulation pipeline.
        The class methods largely produce AST nodes/trees.
        """

        def __init__(self):
            self.__state = 0

        ids = [chr(ord(v) + x) for v in ['a', 'A'] for x in range(26)]

        def varidx(self):
            """
            a monotonically increasing index for use in labelling variables.
            """
            tmp = self.__state
            self.__state = self.__state + 1
            return tmp

        # builder functions
        def gen_alloc_return(self, orig, var, dtype_var, init_val=0):
            """
            Generates an AST equivalent to:
                `var = np.full(orig.shape, init_val, dtype = dtype_var)`
            """
            new = ast.Assign(
                targets=[
                    ast.Name(
                        id=var,
                        ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(
                            id='np',
                            ctx=ast.Load()),
                        attr='full',
                        ctx=ast.Load()),
                    args=[
                        ast.Attribute(
                            value=ast.Name(
                                id=orig,
                                ctx=ast.Load()),
                            attr='shape',
                            ctx=ast.Load()),
                        self.gen_num(init_val)],
                    keywords=[ast.keyword(arg='dtype',
                                          value=self.gen_call('type', [dtype_var.id]).value)],
                    starargs=None,
                    kwargs=None),
            )
            return new

        def gen_assign(self, var, value, index_names):
            """
            Generates an AST equivalent to:
                `retvar[(*index_names,)] = value[<already present indexing>]`
            """
            elts_info = [ast.Name(id=x, ctx=ast.Load()) for x in index_names]
            new = ast.Assign(
                targets=[
                    ast.Subscript(
                        value=ast.Name(
                            id=var,
                            ctx=ast.Load()),
                        slice=ast.Index(
                            value=ast.Tuple(
                                elts=elts_info,
                                ctx=ast.Load())),
                        ctx=ast.Store())],
                value=value)
            return new

        def gen_loop(self, var, start=0, stop=0, body=None):
            """
            Generates an AST equivalent to a loop in `var` from
            `start` to `stop` with body `body`.
            """
            if isinstance(start, int):
                start_val = ast.Num(n=start)
            else:
                start_val = start
            if isinstance(stop, int):
                stop_val = ast.Num(n=stop)
            else:
                stop_val = stop
            return ast.For(
                target=ast.Name(id=var, ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id='range', ctx=ast.Load()),
                    args=[start_val, stop_val],
                    keywords=[],
                    starargs=None, kwargs=None),
                body=body, orelse=[])

        def gen_return(self, var):
            """
            Generates an AST equivalent to `return var`
            """
            return ast.Return(value=ast.Name(id=var, ctx=ast.Load()))

        def gen_slice(self, value):
            """Generates an Index with the given value"""
            return ast.Index(value=ast.Num(n=value))

        def gen_attr(self, name, attr):
            """
            Generates AST equivalent to `name.attr`
            """
            return ast.Attribute(
                value=ast.Name(id=name, ctx=ast.Load()),
                attr=attr, ctx=ast.Load())

        def gen_subscript(self, name, attr, index, offset=None):
            """
            Generates an AST equivalent to a subscript, something like:
                name.attr[slice(index) +/- offset]
            """
            attribute = self.gen_attr(name, attr)
            slise = self.gen_slice(index)
            ss = ast.Subscript(value=attribute, slice=slise, ctx=ast.Load())
            if offset:
                pm = ast.Add() if offset >= 0 else ast.Sub()
                ss = ast.BinOp(left=ss, op=pm, right=ast.Num(n=abs(offset)))
            return ss

        def gen_num(self, value):
            """
            Generates an ast.Num of value `value`
            """
            if abs(value) >= 0:
                return ast.Num(value)
            else:
                return ast.UnaryOp(ast.USub(), ast.Num(-value))

        def gen_call(self, call_name, args, kwargs=None):
            """
            Generates an AST equivalent to a call, something like:
                `call_name(*args, **kwargs)
            """
            fixed_args = [ast.Name(id='%s' % x, ctx=ast.Load()) for x in args]
            if kwargs is not None:
                keywords = [ast.keyword(
                            arg='%s' %
                            x, value=ast.parse(str(x)).body[0].value)
                            for x in kwargs]
            else:
                keywords = []
            func = ast.Name(id=call_name, ctx=ast.Load())
            return ast.Expr(value=ast.Call(
                            func=func, args=fixed_args,
                            keywords=keywords,
                            starargs=None, kwargs=None), ctx=ast.Load())

    # AST transformers
    class FoldConst(ast.NodeTransformer, Builder):
        """
        Folds const expr, this is so const expressions in the relidx are
        more easily handled
        """

        # just support a few for testing purposes
        supported_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
        }

        def visit_BinOp(self, node):
            # does const expr folding
            node = self.generic_visit(node)

            op = self.supported_ops.get(node.op.__class__)
            lhs = getattr(node, 'left', None)
            rhs = getattr(node, 'right', None)

            if not (lhs and rhs and op):
                return node

            if (isinstance(lhs, ast.Num) and
                    isinstance(rhs, ast.Num)):
                return ast.Num(op(node.left.n, node.right.n))
            else:
                return node

    class FixRelIndex(ast.NodeTransformer, Builder):
        """ Fixes the relative indexes to be written in as
        induction index + relative index
        """

        def __init__(self, argnames, const_assigns,
                     standard_indexing, neighborhood, *args, **kwargs):
            ast.NodeTransformer.__init__(self, *args, **kwargs)
            pyStencilGenerator.Builder.__init__(self, *args, **kwargs)
            self._argnames = argnames
            self._const_assigns = const_assigns
            self._idx_len = -1
            self._mins = None
            self._maxes = None
            self._imin = np.iinfo(int).min
            self._imax = np.iinfo(int).max
            self._standard_indexing = standard_indexing \
                if standard_indexing else []
            self._neighborhood = neighborhood
            self._id_pat = '__%sn' if neighborhood else '__%s'

        def get_val_from_num(self, node):
            """
            Gets the literal value from a Num or UnaryOp
            """
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.UnaryOp):
                return -node.operand.n
            else:
                raise ValueError(
                    "get_val_from_num: Unknown indexing operation")

        def visit_Subscript(self, node):
            """
            Transforms subscripts of the form `a[x]` and `a[x, y, z, ...]`
            where `x, y, z` are relative indexes, to forms such as:
            `a[x + i]` and `a[x + i, y + j, z + k]` for use in loop induced
            indexing.
            """

            node = self.generic_visit(node)
            if (node.value.id in self._argnames) and (
                    node.value.id not in self._standard_indexing):
                # 2D index
                if isinstance(node.slice.value, ast.Tuple):
                    idx = []
                    for x, val in enumerate(node.slice.value.elts):
                        useval = self._const_assigns.get(val, val)
                        idx.append(
                            ast.BinOp(
                                left=ast.Name(
                                    id=self._id_pat %
                                    self.ids[x],
                                    ctx=ast.Load()),
                                op=ast.Add(),
                                right=useval,
                                ctx=ast.Load()))
                    if self._idx_len == -1:
                        self._idx_len = len(idx)
                    else:
                        if(self._idx_len != len(idx)):
                            raise ValueError(
                                "Relative indexing mismatch detected")
                    if isinstance(node.ctx, ast.Store):
                        msg = ("Assignments to array passed to "
                               "stencil kernels is not allowed")
                        raise ValueError(msg)
                    context = ast.Load()
                    newnode = ast.Subscript(
                        value=node.value,
                        slice=ast.Index(
                            value=ast.Tuple(
                                elts=idx,
                                ctx=ast.Load()),
                            ctx=ast.Load()),
                        ctx=context)
                    ast.copy_location(newnode, node)
                    ast.fix_missing_locations(newnode)

                    # now work out max/min for index ranges i.e. stencil size
                    if self._mins is None and self._maxes is None:
                        # first pass
                        self._mins = [self._imax] * self._idx_len
                        self._maxes = [self._imin] * self._idx_len

                    if not self._neighborhood:
                        for x, lnode in enumerate(node.slice.value.elts):
                            if isinstance(lnode, ast.Num) or\
                                    isinstance(lnode, ast.UnaryOp):
                                relvalue = self.get_val_from_num(lnode)
                            elif (hasattr(lnode, 'id') and
                                  lnode.id in self._const_assigns):
                                relvalue = self._const_assigns[lnode.id]
                            else:
                                raise ValueError(
                                    "Cannot interpret indexing value")
                            if relvalue < self._mins[x]:
                                self._mins[x] = relvalue
                            if relvalue > self._maxes[x]:
                                self._maxes[x] = relvalue
                    else:
                        for x, lnode in enumerate(self._neighborhood):
                            self._mins[x] = self._neighborhood[x][0]
                            self._maxes[x] = self._neighborhood[x][1]

                    return newnode
                # 1D index
                elif isinstance(node.slice, ast.Index):
                    useval = self._const_assigns.get(
                        node.slice.value, node.slice.value)
                    idx = ast.BinOp(left=ast.Name(
                                    id=self._id_pat %
                                    self.ids[0],
                                    ctx=ast.Load()),
                                    op=ast.Add(),
                                    right=useval,
                                    ctx=ast.Load())
                    if self._idx_len == -1:
                        self._idx_len = 1
                    else:
                        if(self._idx_len != 1):
                            raise ValueError(
                                "Relative indexing mismatch detected")
                    if isinstance(node.ctx, ast.Store):
                        msg = ("Assignments to array passed to "
                               "stencil kernels is not allowed")
                        raise ValueError(msg)
                    context = ast.Load()
                    newnode = ast.Subscript(
                        value=node.value,
                        slice=ast.Index(
                            value=idx,
                            ctx=ast.Load()),
                        ctx=context)
                    ast.copy_location(newnode, node)
                    ast.fix_missing_locations(newnode)

                    # now work out max/min for index ranges i.e. stencil size
                    if self._mins is None and self._maxes is None:
                        # first pass
                        self._mins = [self._imax, ]
                        self._maxes = [self._imin, ]

                    if not self._neighborhood:
                        if isinstance(node.slice.value, ast.Num) or\
                                isinstance(node.slice.value, ast.UnaryOp):
                            relvalue = self.get_val_from_num(node.slice.value)
                        elif (hasattr(node.slice.value, 'id') and
                              node.slice.value.id in self._const_assigns):
                            relvalue = self._const_assigns[node.slice.value.id]
                        else:
                            raise ValueError("Cannot interpret indexing value")
                        if relvalue < self._mins[0]:
                            self._mins[0] = relvalue
                        if relvalue > self._maxes[0]:
                            self._maxes[0] = relvalue
                    else:
                        self._mins[0] = self._neighborhood[0][0]
                        self._maxes[0] = self._neighborhood[0][1]

                    return newnode
                else:  # unknown
                    raise ValueError("Unhandled subscript")
            else:
                return node

        @property
        def idx_len(self):
            if self._idx_len == -1:
                raise ValueError(
                    'Transform has not been run/no indexes found')
            else:
                return self._idx_len

        @property
        def maxes(self):
            return self._maxes

        @property
        def mins(self):
            return self._mins

        @property
        def id_pattern(self):
            return self._id_pat

    class TransformReturns(ast.NodeTransformer, Builder):
        """
        Transforms return nodes into assignments.
        """

        def __init__(self, relidx_info, *args, **kwargs):
            ast.NodeTransformer.__init__(self, *args, **kwargs)
            pyStencilGenerator.Builder.__init__(self, *args, **kwargs)
            self._relidx_info = relidx_info
            self._ret_var_idx = self.varidx()
            retvar = '__b%s' % self._ret_var_idx
            self._retvarname = retvar

        def visit_Return(self, node):
            self.generic_visit(node)
            nloops = self._relidx_info.idx_len
            var_pattern = self._relidx_info.id_pattern
            return self.gen_assign(
                self._retvarname, node.value,
                [var_pattern % self.ids[l] for l in range(nloops)])

        @property
        def ret_var_name(self):
            return self._retvarname

    class FixFunc(ast.NodeTransformer, Builder):
        """ The main function rewriter, takes the body of the kernel and generates:
         * checking function calls
         * return value allocation
         * loop nests
         * return site
         * Function definition as an entry point
        """

        def __init__(self, kprops, relidx_info, ret_info,
                     cval, standard_indexing, neighborhood, *args, **kwargs):
            ast.NodeTransformer.__init__(self, *args, **kwargs)
            pyStencilGenerator.Builder.__init__(self, *args, **kwargs)
            self._original_kernel = kprops.original_kernel
            self._argnames = kprops.argnames
            self._retty = kprops.retty
            self._relidx_info = relidx_info
            self._ret_info = ret_info
            self._standard_indexing = standard_indexing \
                if standard_indexing else []
            self._neighborhood = neighborhood if neighborhood else tuple()
            self._relidx_args = [
                x for x in self._argnames if x not in self._standard_indexing]
            # switch cval to python type
            if hasattr(cval, 'dtype'):
                self.cval = cval.tolist()
            else:
                self.cval = cval
            self.stencil_arr = self._argnames[0]

        def visit_FunctionDef(self, node):
            """
            Transforms the kernel function into a function that will perform
            the stencil like behaviour on the kernel.
            """
            self.generic_visit(node)

            # this function validates arguments and is injected into the top
            # of the stencil call
            def check_stencil_arrays(*args, **kwargs):
                # the first has to be an array due to parfors requirements
                neighborhood = kwargs.get('neighborhood')
                init_shape = args[0].shape
                if neighborhood is not None:
                    if len(init_shape) != len(neighborhood):
                        raise ValueError("Invalid neighborhood supplied")
                for x in args[1:]:
                    if hasattr(x, 'shape'):
                        if init_shape != x.shape:
                            raise ValueError(
                                "Input stencil arrays do not commute")

            checksrc = inspect.getsource(check_stencil_arrays)
            check_impl = ast.parse(
                checksrc.strip()).body[0]  # don't need module
            ast.fix_missing_locations(check_impl)

            checker_call = self.gen_call(
                'check_stencil_arrays',
                self._relidx_args,
                kwargs=['neighborhood'])

            nloops = self._relidx_info.idx_len

            def computebound(mins, maxs):
                minlim = 0 if mins >= 0 else -mins
                maxlim = -maxs if maxs > 0 else 0
                return (minlim, maxlim)

            var_pattern = self._relidx_info.id_pattern

            loop_body = node.body

            # create loop nests
            loop_count = 0
            for l in range(nloops):
                minlim, maxlim = computebound(
                    self._relidx_info.mins[loop_count],
                    self._relidx_info.maxes[loop_count])
                minbound = minlim
                maxbound = self.gen_subscript(
                    self.stencil_arr, 'shape', loop_count, maxlim)
                loops = self.gen_loop(
                    var_pattern % self.ids[loop_count],
                    minbound, maxbound, body=loop_body)
                loop_body = [loops]
                loop_count += 1

            # patch loop location
            ast.copy_location(loops, node)
            _rettyname = self._retty.targets[0]

            # allocate a return
            retvar = self._ret_info.ret_var_name
            allocate = self.gen_alloc_return(
                self.stencil_arr, retvar, _rettyname, self.cval)
            ast.copy_location(allocate, node)

            # generate the return
            returner = self.gen_return(retvar)
            ast.copy_location(returner, node)

            if _py27:
                add_kwarg = [ast.Name('neighborhood', ast.Param())]
            else:
                add_kwarg = [ast.arg('neighborhood', None)]
            defaults = [ast.Name(id='None', ctx=ast.Load())]

            newargs = ast.arguments(
                args=node.args.args +
                add_kwarg,
                defaults=defaults,
                vararg=None,
                kwarg=None,
                kwonlyargs=[],
                kw_defaults=[])
            new = ast.FunctionDef(
                name='__%s' %
                node.name,
                args=newargs,
                body=[
                    check_impl,
                    checker_call,
                    self._original_kernel,
                    self._retty,
                    allocate,
                    loops,
                    returner],
                decorator_list=[])
            ast.copy_location(new, node)
            return new

    class GetKernelProps(ast.NodeVisitor, Builder):
        """ Gets the argument names and other properties
        of the original kernel.
        """

        def __init__(self, *args, **kwargs):
            ast.NodeVisitor.__init__(self, *args, **kwargs)
            pyStencilGenerator.Builder.__init__(self, *args, **kwargs)
            self._argnames = None
            self._kwargnames = None
            self._retty = None
            self._original_kernel = None
            self._const_assigns = {}

        def visit_FunctionDef(self, node):
            if self._argnames is not None or self._kwargnames is not None:
                raise RuntimeError("multiple definition of function/args?")

            if _py27:
                attr = 'id'
            else:
                attr = 'arg'

            self._argnames = [getattr(x, attr) for x in node.args.args]
            if node.args.kwarg:
                self._kwargnames = [x.arg for x in node.args.kwarg]
            compute_retdtype = self.gen_call(node.name, self._argnames)
            self._retty = ast.Assign(targets=[ast.Name(
                id='__retdtype',
                ctx=ast.Store())], value=compute_retdtype.value)
            self._original_kernel = ast.fix_missing_locations(deepcopy(node))
            self.generic_visit(node)

        def visit_Assign(self, node):
            self.generic_visit(node)
            tgt = node.targets
            if len(tgt) == 1:
                target = tgt[0]
                if isinstance(target, ast.Name):
                    if isinstance(node.value, ast.Num):
                        self._const_assigns[target.id] = node.value.n
                    elif isinstance(node.value, ast.UnaryOp):
                        if isinstance(node.value, ast.UAdd):
                            self._const_assigns[target.id] = node.value.n
                        else:
                            self._const_assigns[target.id] = -node.value.n

        @property
        def argnames(self):
            """
            The names of the arguments to the function
            """
            return self._argnames

        @property
        def const_assigns(self):
            """
            A map of variable name to constant for variables that are simple
            constant assignments
            """
            return self._const_assigns

        @property
        def retty(self):
            """
            The return type
            """
            return self._retty

        @property
        def original_kernel(self):
            """
            The original unmutated kernel
            """
            return self._original_kernel

    class FixCalls(ast.NodeTransformer):
        """ Fixes call sites for astor (in case it is in use) """

        def visit_Call(self, node):
            self.generic_visit(node)
            # Add in starargs and kwargs to calls
            new = ast.Call(
                func=node.func,
                args=node.args,
                keywords=node.keywords,
                starargs=None,
                kwargs=None)
            return new

    def generate_stencil_tree(
            self, func, cval, standard_indexing, neighborhood):
        """
        Generates the AST tree for a stencil from:
        func - a python stencil kernel
        cval, standard_indexing and neighborhood as per the @stencil decorator
        """
        src = inspect.getsource(func)
        tree = ast.parse(src.strip())

        # Prints debugging information if True.
        # If astor is installed the decompilation of the AST is also printed
        DEBUG = False
        if DEBUG:
            print("ORIGINAL")
            print(ast.dump(tree))

        def pipeline(tree):
            """ the pipeline of manipulations """

            # get the arg names
            kernel_props = self.GetKernelProps()
            kernel_props.visit(tree)
            argnm = kernel_props.argnames
            const_asgn = kernel_props.const_assigns

            if standard_indexing:
                for x in standard_indexing:
                    if x not in argnm:
                        msg = ("Non-existent variable "
                               "specified in standard_indexing")
                        raise ValueError(msg)

            # fold consts
            fold_const = self.FoldConst()
            fold_const.visit(tree)

            # rewrite the relative indices as induced indices
            relidx_fixer = self.FixRelIndex(
                argnm, const_asgn, standard_indexing, neighborhood)
            relidx_fixer.visit(tree)

            # switch returns into assigns
            return_transformer = self.TransformReturns(relidx_fixer)
            return_transformer.visit(tree)

            # generate the function body and loop nests and assemble
            fixer = self.FixFunc(
                kernel_props,
                relidx_fixer,
                return_transformer,
                cval,
                standard_indexing,
                neighborhood)
            fixer.visit(tree)

            # fix up the call sites so they work better with astor
            callFixer = self.FixCalls()
            callFixer.visit(tree)
            ast.fix_missing_locations(tree.body[0])

        # run the pipeline of transforms on the tree
        pipeline(tree)

        if DEBUG:
            print("\n\n\nNEW")
            print(ast.dump(tree, include_attributes=True))
            try:
                import astor
                print(astor.to_source(tree))
            except ImportError:
                pass

        return tree


def pyStencil(func_or_mode='constant', **options):
    """
    A pure python implementation of (a large subset of) stencil functionality,
    equivalent to StencilFunc.
    """

    if not isinstance(func_or_mode, str):
        mode = 'constant'  # default style
        func = func_or_mode
    else:
        assert isinstance(func_or_mode, str), """stencil mode should be
                                                        a string"""
        mode = func_or_mode
        func = None

    for option in options:
        if option not in ["cval", "standard_indexing", "neighborhood"]:
            raise ValueError("Unknown stencil option " + option)

    if mode != 'constant':
        raise ValueError("Unsupported mode style " + mode)

    cval = options.get('cval', 0)
    standard_indexing = options.get('standard_indexing', None)
    neighborhood = options.get('neighborhood', None)

    # generate a new AST tree from the kernel func
    gen = pyStencilGenerator()
    tree = gen.generate_stencil_tree(func, cval, standard_indexing,
                                     neighborhood)

    # breathe life into the tree
    mod_code = compile(tree, filename="<ast>", mode="exec")
    func_code = mod_code.co_consts[0]
    full_func = pytypes.FunctionType(func_code, globals())

    return full_func


@skip_unsupported
class TestManyStencils(TestStencilBase):

    def __init__(self, *args, **kwargs):
        super(TestManyStencils, self).__init__(*args, **kwargs)

    def check(self, pyfunc, *args, **kwargs):
        """
        For a given kernel:

        The expected result is computed from a pyStencil version of the
        stencil.

        The following results are then computed:
        * from a pure @stencil decoration of the kernel.
        * from the njit of a trivial wrapper function around the pure @stencil
          decorated function.
        * from the njit(parallel=True) of a trivial wrapper function around
           the pure @stencil decorated function.

        The results are then compared.
        """

        options = kwargs.get('options', dict())
        expected_exception = kwargs.get('expected_exception')

        # DEBUG print output arrays
        DEBUG_OUTPUT = False

        # collect fails
        should_fail = []
        should_not_fail = []

        # runner that handles fails
        @contextmanager
        def errorhandler(exty=None, usecase=None):
            try:
                yield
            except Exception as e:
                if exty is not None:
                    lexty = exty if hasattr(exty, '__iter__') else [exty, ]
                    found = False
                    for ex in lexty:
                        found |= isinstance(e, ex)
                    if not found:
                        raise
                else:
                    should_not_fail.append(
                        (usecase, "%s: %s" %
                         (type(e), str(e))))
            else:
                if exty is not None:
                    should_fail.append(usecase)

        if isinstance(expected_exception, dict):
            pystencil_ex = expected_exception['pyStencil']
            stencil_ex = expected_exception['stencil']
            njit_ex = expected_exception['njit']
            parfor_ex = expected_exception['parfor']
        else:
            pystencil_ex = expected_exception
            stencil_ex = expected_exception
            njit_ex = expected_exception
            parfor_ex = expected_exception

        stencil_args = {'func_or_mode': pyfunc}
        stencil_args.update(options)

        expected_present = True
        try:
            # ast impl
            ast_impl = pyStencil(func_or_mode=pyfunc, **options)
            expected = ast_impl(
                *args, neighborhood=options.get('neighborhood'))
            if DEBUG_OUTPUT:
                print("\nExpected:\n", expected)
        except Exception as ex:
            # check exception is expected
            with errorhandler(pystencil_ex, "pyStencil"):
                raise ex
            pyStencil_unhandled_ex = ex
            expected_present = False
        stencilfunc_output = None
        with errorhandler(stencil_ex, "@stencil"):
            stencil_func_impl = stencil(**stencil_args)
            # stencil result
            stencilfunc_output = stencil_func_impl(*args)

        # wrapped stencil impl, could this be generated?
        if len(args) == 1:
            def wrap_stencil(arg0):
                return stencil_func_impl(arg0)
        elif len(args) == 2:
            def wrap_stencil(arg0, arg1):
                return stencil_func_impl(arg0, arg1)
        elif len(args) == 3:
            def wrap_stencil(arg0, arg1, arg2):
                return stencil_func_impl(arg0, arg1, arg2)
        else:
            raise ValueError(
                "Up to 3 arguments can be provided, found %s" %
                len(args))

        sig = tuple([numba.typeof(x) for x in args])

        njit_output = None
        with errorhandler(njit_ex, "njit"):
            wrapped_cfunc = self.compile_njit(wrap_stencil, sig)
            # njit result
            njit_output = wrapped_cfunc.entry_point(*args)

        parfor_output = None
        with errorhandler(parfor_ex, "parfors"):
            wrapped_cpfunc = self.compile_parallel(wrap_stencil, sig)
            # parfor result
            parfor_output = wrapped_cpfunc.entry_point(*args)

        if DEBUG_OUTPUT:
            print("\n@stencil_output:\n", stencilfunc_output)
            print("\nnjit_output:\n", njit_output)
            print("\nparfor_output:\n", parfor_output)

        if expected_present:
            try:
                if not stencil_ex:
                    np.testing.assert_almost_equal(
                        stencilfunc_output, expected, decimal=1)
                    self.assertEqual(expected.dtype, stencilfunc_output.dtype)
            except Exception as e:
                should_not_fail.append(
                    ('@stencil', "%s: %s" %
                     (type(e), str(e))))
                print("@stencil failed: %s" % str(e))

            try:
                if not njit_ex:
                    np.testing.assert_almost_equal(
                        njit_output, expected, decimal=1)
                    self.assertEqual(expected.dtype, njit_output.dtype)
            except Exception as e:
                should_not_fail.append(('njit', "%s: %s" % (type(e), str(e))))
                print("@njit failed: %s" % str(e))

            try:
                if not parfor_ex:
                    np.testing.assert_almost_equal(
                        parfor_output, expected, decimal=1)
                    self.assertEqual(expected.dtype, parfor_output.dtype)
                    try:
                        self.assertIn(
                            '@do_scheduling',
                            wrapped_cpfunc.library.get_llvm_str())
                    except AssertionError:
                        msg = 'Could not find `@do_scheduling` in LLVM IR'
                        raise AssertionError(msg)
            except Exception as e:
                should_not_fail.append(
                    ('parfors', "%s: %s" %
                     (type(e), str(e))))
                print("@njit(parallel=True) failed: %s" % str(e))

        if DEBUG_OUTPUT:
            print("\n\n")

        if should_fail:
            msg = ["%s" % x for x in should_fail]
            raise RuntimeError(("The following implementations should have "
                                "raised an exception but did not:\n%s") % msg)

        if should_not_fail:
            impls = ["%s" % x[0] for x in should_not_fail]
            errs = ''.join(["%s: Message: %s\n\n" %
                            x for x in should_not_fail])
            str1 = ("The following implementations should not have raised an "
                    "exception but did:\n%s\n" % impls)
            str2 = "Errors were:\n\n%s" % errs
            raise RuntimeError(str1 + str2)

        if not expected_present:
            if expected_exception is None:
                raise RuntimeError(
                    "pyStencil failed, was not caught/expected",
                    pyStencil_unhandled_ex)

    def exception_dict(self, **kwargs):
        d = dict()
        d['pyStencil'] = None
        d['stencil'] = None
        d['njit'] = None
        d['parfor'] = None
        for k, v in kwargs.items():
            d[k] = v
        return d

    def test_basic00(self):
        """rel index"""
        def kernel(a):
            return a[0, 0]
        a = np.arange(12).reshape(3, 4)
        self.check(kernel, a)

    def test_basic01(self):
        """rel index add const"""
        def kernel(a):
            return a[0, 1]
        a = np.arange(12.).reshape(3, 4)
        self.check(kernel, a)

    def test_basic02(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[0, -1]
        self.check(kernel, a)

    def test_basic03(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[1, 0]
        self.check(kernel, a)

    def test_basic04(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-1, 0]
        self.check(kernel, a)

    def test_basic05(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-1, 1]
        self.check(kernel, a)

    def test_basic06(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[1, -1]
        self.check(kernel, a)

    def test_basic07(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[1, 1]
        self.check(kernel, a)

    def test_basic08(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-1, -1]
        self.check(kernel, a)

    def test_basic09(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-2, 2]
        self.check(kernel, a)

    def test_basic10(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[1, 0]
        self.check(kernel, a)

    def test_basic11(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-1, 0] + a[1, 0]
        self.check(kernel, a)

    def test_basic12(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-1, 1] + a[1, -1]
        self.check(kernel, a)

    def test_basic13(self):
        """rel index add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-1, -1] + a[1, 1]
        self.check(kernel, a)

    def test_basic14(self):
        """rel index add domain change const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + 1j
        self.check(kernel, a)

    def test_basic14b(self):
        """rel index add domain change const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            t = 1.j
            return a[0, 0] + t
        self.check(kernel, a)

    def test_basic15(self):
        """two rel index, add const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[1, 0] + 1.
        self.check(kernel, a)

    def test_basic16(self):
        """two rel index OOB, add const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[10, 0] + 1.

        # only pyStencil bounds checks
        ex = self.exception_dict(pyStencil=IndexError)
        self.check(kernel, a, expected_exception=ex)

    def test_basic17(self):
        """two rel index boundary test, add const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[2, 0] + 1.
        self.check(kernel, a)

    def test_basic18(self):
        """two rel index boundary test, add const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[-2, 0] + 1.
        self.check(kernel, a)

    def test_basic19(self):
        """two rel index boundary test, add const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[0, 3] + 1.
        self.check(kernel, a)

    def test_basic20(self):
        """two rel index boundary test, add const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[0, -3] + 1.
        self.check(kernel, a)

    def test_basic21(self):
        """same rel, add const"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + a[0, 0] + 1.
        self.check(kernel, a)

    def test_basic22(self):
        """rel idx const expr folding, add const"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[1 + 0, 0] + a[0, 0] + 1.
        self.check(kernel, a)

    def test_basic23(self):
        """rel idx, work in body"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            x = np.sin(10 + a[2, 1])
            return a[1 + 0, 0] + a[0, 0] + x
        self.check(kernel, a)

    def test_basic23a(self):
        """rel idx, dead code should not impact rel idx"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            x = np.sin(10 + a[2, 1])
            return a[1 + 0, 0] + a[0, 0]
        self.check(kernel, a)

    def test_basic24(self):
        """1d idx on 2d arr"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return a[0] + 1.
        self.check(kernel, a, expected_exception=[ValueError, TypingError])

    def test_basic25(self):
        """no idx on 2d arr"""
        a = np.arange(12).reshape(3, 4)

        def kernel(a):
            return 1.
        self.check(kernel, a, expected_exception=[ValueError, LoweringError])

    def test_basic26(self):
        """3d arr"""
        a = np.arange(64).reshape(4, 8, 2)

        def kernel(a):
            return a[0, 0, 0] - a[0, 1, 0] + 1.
        self.check(kernel, a)

    def test_basic27(self):
        """4d arr"""
        a = np.arange(128).reshape(4, 8, 2, 2)

        def kernel(a):
            return a[0, 0, 0, 0] - a[0, 1, 0, -1] + 1.
        self.check(kernel, a)

    def test_basic28(self):
        """type widen """
        a = np.arange(12).reshape(3, 4).astype(np.float32)

        def kernel(a):
            return a[0, 0] + np.float64(10.)
        self.check(kernel, a)

    def test_basic29(self):
        """const index from func """
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[0, int(np.cos(0))]
        self.check(kernel, a, expected_exception=[ValueError, LoweringError])

    def test_basic30(self):
        """signed zeros"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[-0, -0]
        self.check(kernel, a)

    def test_basic31(self):
        """does a const propagate? 2D"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            t = 1
            return a[t, 0]
        self.check(kernel, a)

    @unittest.skip("constant folding not implemented")
    def test_basic31b(self):
        """does a const propagate?"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            s = 1
            t = 1 - s
            return a[t, 0]
        self.check(kernel, a)

    def test_basic31c(self):
        """does a const propagate? 1D"""
        a = np.arange(12.)

        def kernel(a):
            t = 1
            return a[t]
        self.check(kernel, a)

    def test_basic32(self):
        """typed int index"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[np.int8(1), 0]
        self.check(kernel, a, expected_exception=[ValueError, LoweringError])

    def test_basic33(self):
        """add 0d array"""
        a = np.arange(12.).reshape(3, 4)

        def kernel(a):
            return a[0, 0] + np.array(1)
        self.check(kernel, a)

    def test_basic34(self):
        """More complex rel index with dependency on addition rel index"""
        def kernel(a):
            g = 4. + a[0, 1]
            return g + (a[0, 1] + a[1, 0] + a[0, -1] + np.sin(a[-2, 0]))
        a = np.arange(144).reshape(12, 12)
        self.check(kernel, a)

    def test_basic35(self):
        """simple cval """
        def kernel(a):
            return a[0, 1]
        a = np.arange(12.).reshape(3, 4)
        ex = self.exception_dict(
            stencil=ValueError,
            parfor=ValueError,
            njit=LoweringError)
        self.check(kernel, a, options={'cval': 5}, expected_exception=ex)

    def test_basic36(self):
        """more complex with cval"""
        def kernel(a):
            return a[0, 1] + a[0, -1] + a[1, -1] + a[1, -1]
        a = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, options={'cval': 5.})

    def test_basic37(self):
        """cval is expr"""
        def kernel(a):
            return a[0, 1] + a[0, -1] + a[1, -1] + a[1, -1]
        a = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, options={'cval': 5 + 63.})

    def test_basic38(self):
        """cval is complex"""
        def kernel(a):
            return a[0, 1] + a[0, -1] + a[1, -1] + a[1, -1]
        a = np.arange(12.).reshape(3, 4)
        ex = self.exception_dict(
            stencil=ValueError,
            parfor=ValueError,
            njit=LoweringError)
        self.check(kernel, a, options={'cval': 1.j}, expected_exception=ex)

    def test_basic39(self):
        """cval is func expr"""
        def kernel(a):
            return a[0, 1] + a[0, -1] + a[1, -1] + a[1, -1]
        a = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, options={'cval': np.sin(3.) + np.cos(2)})

    def test_basic40(self):
        """2 args!"""
        def kernel(a, b):
            return a[0, 1] + b[0, -2]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b)

    def test_basic41(self):
        """2 args! rel arrays wildly not same size!"""
        def kernel(a, b):
            return a[0, 1] + b[0, -2]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(1.).reshape(1, 1)
        self.check(
            kernel, a, b, expected_exception=[
                ValueError, AssertionError])

    def test_basic42(self):
        """2 args! rel arrays very close in size"""
        def kernel(a, b):
            return a[0, 1] + b[0, -2]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(9.).reshape(3, 3)
        self.check(
            kernel, a, b, expected_exception=[
                ValueError, AssertionError])

    def test_basic43(self):
        """2 args more complexity"""
        def kernel(a, b):
            return a[0, 1] + a[1, 2] + b[-2, 0] + b[0, -1]
        a = np.arange(30.).reshape(5, 6)
        b = np.arange(30.).reshape(5, 6)
        self.check(kernel, a, b)

    def test_basic44(self):
        """2 args, has assignment before use"""
        def kernel(a, b):
            a[0, 1] = 12
            return a[0, 1]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(
            kernel, a, b, expected_exception=[
                ValueError, LoweringError])

    def test_basic45(self):
        """2 args, has assignment and then cross dependency"""
        def kernel(a, b):
            a[0, 1] = 12
            return a[0, 1] + a[1, 0]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(
            kernel, a, b, expected_exception=[
                ValueError, LoweringError])

    def test_basic46(self):
        """2 args, has cross relidx assignment"""
        def kernel(a, b):
            a[0, 1] = b[1, 2]
            return a[0, 1] + a[1, 0]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(
            kernel, a, b, expected_exception=[
                ValueError, LoweringError])

    def test_basic47(self):
        """3 args"""
        def kernel(a, b, c):
            return a[0, 1] + b[1, 0] + c[-1, 0]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        c = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, c)

    # matches pyStencil, but all ought to fail
    # probably hard to detect?
    def test_basic48(self):
        """2 args, has assignment before use via memory alias"""
        def kernel(a):
            c = a.T
            c[:, :] = 10
            return a[0, 1]
        a = np.arange(12.).reshape(3, 4)
        self.check(kernel, a)

    def test_basic49(self):
        """2 args, standard_indexing on second"""
        def kernel(a, b):
            return a[0, 1] + b[0, 3]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'b'})

    @unittest.skip("dynamic range checking not implemented")
    def test_basic50(self):
        """2 args, standard_indexing OOB"""
        def kernel(a, b):
            return a[0, 1] + b[0, 15]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(
            kernel,
            a,
            b,
            options={
                'standard_indexing': 'b'},
            expected_exception=IndexError)

    def test_basic51(self):
        """2 args, standard_indexing, no relidx"""
        def kernel(a, b):
            return a[0, 1] + b[0, 2]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(
            kernel, a, b, options={
                'standard_indexing': [
                    'a', 'b']}, expected_exception=[
                ValueError, LoweringError])

    def test_basic52(self):
        """3 args, standard_indexing on middle arg """
        def kernel(a, b, c):
            return a[0, 1] + b[0, 1] + c[1, 2]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(4.).reshape(2, 2)
        c = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, c, options={'standard_indexing': 'b'})

    def test_basic53(self):
        """2 args, standard_indexing on variable that does not exist"""
        def kernel(a, b):
            return a[0, 1] + b[0, 2]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        ex = self.exception_dict(
            pyStencil=ValueError,
            stencil=Exception,
            parfor=ValueError,
            njit=Exception)
        self.check(
            kernel,
            a,
            b,
            options={
                'standard_indexing': 'c'},
            expected_exception=ex)

    def test_basic54(self):
        """2 args, standard_indexing, index from var"""
        def kernel(a, b):
            t = 2
            return a[0, 1] + b[0, t]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'b'})

    def test_basic55(self):
        """2 args, standard_indexing, index from more complex var"""
        def kernel(a, b):
            s = 1
            t = 2 - s
            return a[0, 1] + b[0, t]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'b'})

    def test_basic56(self):
        """2 args, standard_indexing, added complexity """
        def kernel(a, b):
            s = 1
            acc = 0
            for k in b[0, :]:
                acc += k
            t = 2 - s - 1
            return a[0, 1] + b[0, t] + acc
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'b'})

    def test_basic57(self):
        """2 args, standard_indexing, split index operation """
        def kernel(a, b):
            c = b[0]
            return a[0, 1] + c[1]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'b'})

    def test_basic58(self):
        """2 args, standard_indexing, split index with broadcast mutation """
        def kernel(a, b):
            c = b[0] + 1
            return a[0, 1] + c[1]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(kernel, a, b, options={'standard_indexing': 'b'})

    def test_basic59(self):
        """3 args, mix of array, relative and standard indexing and const"""
        def kernel(a, b, c):
            return a[0, 1] + b[1, 1] + c
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        c = 10
        self.check(kernel, a, b, c, options={'standard_indexing': ['b', 'c']})

    def test_basic60(self):
        """3 args, mix of array, relative and standard indexing,
        tuple pass through"""
        def kernel(a, b, c):
            return a[0, 1] + b[1, 1] + c[0]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        c = (10,)
        # parfors does not support tuple args for stencil kernels
        ex = self.exception_dict(parfor=ValueError)
        self.check(
            kernel, a, b, c, options={
                'standard_indexing': [
                    'b', 'c']}, expected_exception=ex)

    def test_basic61(self):
        """2 args, standard_indexing on first"""
        def kernel(a, b):
            return a[0, 1] + b[1, 1]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(
            kernel,
            a,
            b,
            options={
                'standard_indexing': 'a'},
            expected_exception=Exception)

    def test_basic62(self):
        """2 args, standard_indexing and cval"""
        def kernel(a, b):
            return a[0, 1] + b[1, 1]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12.).reshape(3, 4)
        self.check(
            kernel,
            a,
            b,
            options={
                'standard_indexing': 'b',
                'cval': 10.})

    def test_basic63(self):
        """2 args, standard_indexing applied to relative, should fail,
        non-const idx"""
        def kernel(a, b):
            return a[0, b[0, 1]]
        a = np.arange(12.).reshape(3, 4)
        b = np.arange(12).reshape(3, 4)
        ex = self.exception_dict(
            pyStencil=ValueError,
            stencil=ValueError,
            parfor=ValueError,
            njit=LoweringError)
        self.check(
            kernel,
            a,
            b,
            options={
                'standard_indexing': 'b'},
            expected_exception=ex)

    # stencil, njit, parfors all fail. Does this make sense?
    def test_basic64(self):
        """1 arg that uses standard_indexing"""
        def kernel(a):
            return a[0, 0]
        a = np.arange(12.).reshape(3, 4)
        self.check(
            kernel,
            a,
            options={
                'standard_indexing': 'a'},
            expected_exception=[
                ValueError,
                LoweringError])

    def test_basic65(self):
        """basic induced neighborhood test"""
        def kernel(a):
            cumul = 0
            for i in range(-29, 1):
                cumul += a[i]
            return cumul / 30
        a = np.arange(60.)
        self.check(kernel, a, options={'neighborhood': ((-29, 0),)})

    # Should this work? a[0] is out of neighborhood?
    def test_basic66(self):
        """basic const neighborhood test"""
        def kernel(a):
            cumul = 0
            for i in range(-29, 1):
                cumul += a[0]
            return cumul / 30
        a = np.arange(60.)
        self.check(kernel, a, options={'neighborhood': ((-29, 0),)})

    def test_basic67(self):
        """basic 2d induced neighborhood test"""
        def kernel(a):
            cumul = 0
            for i in range(-5, 1):
                for j in range(-10, 1):
                    cumul += a[i, j]
            return cumul / (10 * 5)
        a = np.arange(10. * 20.).reshape(10, 20)
        self.check(kernel, a, options={'neighborhood': ((-5, 0), (-10, 0),)})

    def test_basic67b(self):
        """basic 2d induced 1D neighborhood"""
        def kernel(a):
            cumul = 0
            for j in range(-10, 1):
                cumul += a[0, j]
            return cumul / (10 * 5)
        a = np.arange(10. * 20.).reshape(10, 20)
        self.check(
            kernel,
            a,
            options={
                'neighborhood': (
                    (-10,
                     0),
                )},
            expected_exception=[
                TypingError,
                ValueError])

    # Should this work or is it UB? a[i, 0] is out of neighborhood?
    def test_basic68(self):
        """basic 2d one induced, one cost neighborhood test"""
        def kernel(a):
            cumul = 0
            for i in range(-5, 1):
                for j in range(-10, 1):
                    cumul += a[i, 0]
            return cumul / (10 * 5)
        a = np.arange(10. * 20.).reshape(10, 20)
        self.check(kernel, a, options={'neighborhood': ((-5, 0), (-10, 0),)})

    # Should this work or is it UB? a[0, 0] is out of neighborhood?
    def test_basic69(self):
        """basic 2d two cost neighborhood test"""
        def kernel(a):
            cumul = 0
            for i in range(-5, 1):
                for j in range(-10, 1):
                    cumul += a[0, 0]
            return cumul / (10 * 5)
        a = np.arange(10. * 20.).reshape(10, 20)
        self.check(kernel, a, options={'neighborhood': ((-5, 0), (-10, 0),)})

    def test_basic70(self):
        """neighborhood adding complexity"""
        def kernel(a):
            cumul = 0
            zz = 12.
            for i in range(-5, 1):
                t = zz + i
                for j in range(-10, 1):
                    cumul += a[i, j] + t
            return cumul / (10 * 5)
        a = np.arange(10. * 20.).reshape(10, 20)
        self.check(kernel, a, options={'neighborhood': ((-5, 0), (-10, 0),)})

    def test_basic71(self):
        """neighborhood, type change"""
        def kernel(a):
            cumul = 0
            for i in range(-29, 1):
                k = 0.
                if i > -15:
                    k = 1j
                cumul += a[i] + k
            return cumul / 30
        a = np.arange(60.)
        self.check(kernel, a, options={'neighborhood': ((-29, 0),)})

    def test_basic72(self):
        """neighborhood, narrower range than specified"""
        def kernel(a):
            cumul = 0
            for i in range(-19, -3):
                cumul += a[i]
            return cumul / 30
        a = np.arange(60.)
        self.check(kernel, a, options={'neighborhood': ((-29, 0),)})

    def test_basic73(self):
        """neighborhood, +ve range"""
        def kernel(a):
            cumul = 0
            for i in range(5, 11):
                cumul += a[i]
            return cumul / 30
        a = np.arange(60.)
        self.check(kernel, a, options={'neighborhood': ((5, 10),)})

    def test_basic73b(self):
        """neighborhood, -ve range"""
        def kernel(a):
            cumul = 0
            for i in range(-10, -4):
                cumul += a[i]
            return cumul / 30
        a = np.arange(60.)
        self.check(kernel, a, options={'neighborhood': ((-10, -5),)})

    def test_basic74(self):
        """neighborhood, -ve->+ve range span"""
        def kernel(a):
            cumul = 0
            for i in range(-5, 11):
                cumul += a[i]
            return cumul / 30
        a = np.arange(60.)
        self.check(kernel, a, options={'neighborhood': ((-5, 10),)})

    def test_basic75(self):
        """neighborhood, -ve->-ve range span"""
        def kernel(a):
            cumul = 0
            for i in range(-10, -1):
                cumul += a[i]
            return cumul / 30
        a = np.arange(60.)
        self.check(kernel, a, options={'neighborhood': ((-10, -2),)})

    def test_basic76(self):
        """neighborhood, mixed range span"""
        def kernel(a):
            cumul = 0
            zz = 12.
            for i in range(-3, 0):
                t = zz + i
                for j in range(-3, 4):
                    cumul += a[i, j] + t
            return cumul / (10 * 5)
        a = np.arange(10. * 20.).reshape(10, 20)
        self.check(kernel, a, options={'neighborhood': ((-3, -1), (-3, 3),)})

    def test_basic77(self):
        """ neighborhood, two args """
        def kernel(a, b):
            cumul = 0
            for i in range(-3, 1):
                for j in range(-3, 1):
                    cumul += a[i, j] + b[i, j]
            return cumul / (9.)
        a = np.arange(10. * 20.).reshape(10, 20)
        b = np.arange(10. * 20.).reshape(10, 20)
        self.check(kernel, a, b, options={'neighborhood': ((-3, 0), (-3, 0),)})

    def test_basic78(self):
        """ neighborhood, two args, -ve range, -ve range """
        def kernel(a, b):
            cumul = 0
            for i in range(-6, -2):
                for j in range(-7, -1):
                    cumul += a[i, j] + b[i, j]
            return cumul / (9.)
        a = np.arange(15. * 20.).reshape(15, 20)
        b = np.arange(15. * 20.).reshape(15, 20)
        self.check(
            kernel, a, b, options={
                'neighborhood': (
                    (-6, -3), (-7, -2),)})

    def test_basic78b(self):
        """ neighborhood, two args, -ve range, +ve range """
        def kernel(a, b):
            cumul = 0
            for i in range(-6, -2):
                for j in range(2, 10):
                    cumul += a[i, j] + b[i, j]
            return cumul / (9.)
        a = np.arange(15. * 20.).reshape(15, 20)
        b = np.arange(15. * 20.).reshape(15, 20)
        self.check(kernel, a, b, options={'neighborhood': ((-6, -3), (2, 9),)})

    def test_basic79(self):
        """ neighborhood, two incompatible args """
        def kernel(a, b):
            cumul = 0
            for i in range(-3, 1):
                for j in range(-3, 1):
                    cumul += a[i, j] + b[i, j]
            return cumul / (9.)
        a = np.arange(10. * 20.).reshape(10, 20)
        b = np.arange(10. * 20.).reshape(10, 10, 2)
        ex = self.exception_dict(
            pyStencil=ValueError,
            stencil=TypingError,
            parfor=TypingError,
            njit=TypingError)
        self.check(
            kernel, a, b, options={
                'neighborhood': (
                    (-3, 0), (-3, 0),)}, expected_exception=ex)

    def test_basic80(self):
        """ neighborhood, type change """
        def kernel(a, b):
            cumul = 0
            for i in range(-3, 1):
                for j in range(-3, 1):
                    cumul += a[i, j] + b
            return cumul / (9.)
        a = np.arange(10. * 20.).reshape(10, 20)
        b = 12.j
        self.check(kernel, a, b, options={'neighborhood': ((-3, 0), (-3, 0))})

    def test_basic81(self):
        """ neighborhood, dimensionally incompatible arrays """
        def kernel(a, b):
            cumul = 0
            for i in range(-3, 1):
                for j in range(-3, 1):
                    cumul += a[i, j] + b[i]
            return cumul / (9.)
        a = np.arange(10. * 20.).reshape(10, 20)
        b = a[0].copy()
        ex = self.exception_dict(
            pyStencil=ValueError,
            stencil=TypingError,
            parfor=AssertionError,
            njit=TypingError)
        self.check(
            kernel, a, b, options={
                'neighborhood': (
                    (-3, 0), (-3, 0))}, expected_exception=ex)

    def test_basic82(self):
        """ neighborhood, with standard_indexing"""
        def kernel(a, b):
            cumul = 0
            for i in range(-3, 1):
                for j in range(-3, 1):
                    cumul += a[i, j] + b[1, 3]
            return cumul / (9.)
        a = np.arange(10. * 20.).reshape(10, 20)
        b = a.copy()
        self.check(
            kernel, a, b, options={
                'neighborhood': (
                    (-3, 0), (-3, 0)), 'standard_indexing': 'b'})

    def test_basic83(self):
        """ neighborhood, with standard_indexing and cval"""
        def kernel(a, b):
            cumul = 0
            for i in range(-3, 1):
                for j in range(-3, 1):
                    cumul += a[i, j] + b[1, 3]
            return cumul / (9.)
        a = np.arange(10. * 20.).reshape(10, 20)
        b = a.copy()
        self.check(
            kernel, a, b, options={
                'neighborhood': (
                    (-3, 0), (-3, 0)), 'standard_indexing': 'b', 'cval': 1.5})

    def test_basic84(self):
        """ kernel calls njit """
        def kernel(a):
            return a[0, 0] + addone_njit(a[0, 1])

        a = np.arange(10. * 20.).reshape(10, 20)
        self.check(kernel, a)

    def test_basic85(self):
        """ kernel calls njit(parallel=True)"""
        def kernel(a):
            return a[0, 0] + addone_pjit(a[0, 1])

        a = np.arange(10. * 20.).reshape(10, 20)
        self.check(kernel, a)

    # njit/parfors fail correctly, but the error message isn't very informative
    def test_basic86(self):
        """ bad kwarg """
        def kernel(a):
            return a[0, 0]

        a = np.arange(10. * 20.).reshape(10, 20)
        self.check(kernel, a, options={'bad': 10},
                   expected_exception=[ValueError, TypingError])

    def test_basic87(self):
        """ reserved arg name in use """
        def kernel(__sentinel__):
            return __sentinel__[0, 0]

        a = np.arange(10. * 20.).reshape(10, 20)
        self.check(kernel, a)

    def test_basic88(self):
        """ use of reserved word """
        def kernel(a, out):
            return out * a[0, 1]
        a = np.arange(12.).reshape(3, 4)
        ex = self.exception_dict(
            pyStencil=ValueError,
            stencil=ValueError,
            parfor=ValueError,
            njit=LoweringError)
        self.check(
            kernel,
            a,
            1.0,
            options={},
            expected_exception=ex)

    def test_basic89(self):
        """ basic multiple return"""
        def kernel(a):
            if a[0, 1] > 10:
                return 10.
            elif a[0, 3] < 8:
                return a[0, 0]
            else:
                return 7.

        a = np.arange(10. * 20.).reshape(10, 20)
        self.check(kernel, a)

    def test_basic90(self):
        """ neighborhood, with standard_indexing and cval, multiple returns"""
        def kernel(a, b):
            cumul = 0
            for i in range(-3, 1):
                for j in range(-3, 1):
                    cumul += a[i, j] + b[1, 3]
            res = cumul / (9.)
            if res > 200.0:
                return res + 1.0
            else:
                return res
        a = np.arange(10. * 20.).reshape(10, 20)
        b = a.copy()
        self.check(
            kernel, a, b, options={
                'neighborhood': (
                    (-3, 0), (-3, 0)), 'standard_indexing': 'b', 'cval': 1.5})


if __name__ == "__main__":
    unittest.main()
