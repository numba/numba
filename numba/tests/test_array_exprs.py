from __future__ import print_function, division, absolute_import
from timeit import Timer

import numpy as np

from numba import njit, vectorize
from numba import unittest_support as unittest
from numba import compiler, typing, typeof, ir
from numba.compiler import Pipeline, _PipelineManager, Flags
from numba.targets import cpu


class Namespace(dict):
    def __getattr__(s, k):
        return s[k] if k in s else super(Namespace, s).__getattr__(k)

def axy(a, x, y):
    return a * x + y

def ax2(a, x, y):
    return a * x + y

def pos_root(As, Bs, Cs):
    return (-Bs + (((Bs ** 2.) - (4. * As * Cs)) ** 0.5)) / (2. * As)

def neg_root_common_subexpr(As, Bs, Cs):
    _2As = 2. * As
    _4AsCs = 2. * _2As * Cs
    _Bs2_4AsCs = (Bs ** 2. - _4AsCs)
    return (-Bs - (_Bs2_4AsCs ** 0.5)) / _2As

def neg_root_complex_subexpr(As, Bs, Cs):
    _2As = 2. * As
    _4AsCs = 2. * _2As * Cs
    _Bs2_4AsCs = (Bs ** 2. - _4AsCs) + 0j # Force into the complex domain.
    return (-Bs - (_Bs2_4AsCs ** 0.5)) / _2As

vaxy = vectorize(axy)

def call_stuff(a0, a1):
    return np.cos(vaxy(a0, np.sin(a1) - 1., 1.))

def are_roots_imaginary(As, Bs, Cs):
    return (Bs ** 2 - 4 * As * Cs) < 0


class RewritesTester(Pipeline):
    @classmethod
    def mk_pipeline(cls, args, return_type=None, flags=None, locals={},
                    library=None, typing_context=None, target_context=None):
        if not flags:
            flags = Flags()
        flags.nrt = True
        if typing_context is None:
            typing_context = typing.Context()
        if target_context is None:
            target_context = cpu.CPUContext(typing_context)
        return cls(typing_context, target_context, library, args, return_type,
                   flags, locals)

    @classmethod
    def mk_no_rw_pipeline(cls, args, return_type=None, flags=None, locals={},
                          library=None, **kws):
        if not flags:
            flags = Flags()
        flags.no_rewrites = True
        return cls.mk_pipeline(args, return_type, flags, locals, library, **kws)


class TestArrayExpressions(unittest.TestCase):

    def test_simple_expr(self):
        '''
        Using a simple array expression, verify that rewriting is taking
        place, and is fusing loops.
        '''
        A = np.linspace(0,1,10)
        X = np.linspace(2,1,10)
        Y = np.linspace(1,2,10)
        arg_tys = [typeof(arg) for arg in (A, X, Y)]

        control_pipeline = RewritesTester.mk_no_rw_pipeline(arg_tys)
        cres_0 = control_pipeline.compile_extra(axy)
        nb_axy_0 = cres_0.entry_point

        test_pipeline = RewritesTester.mk_pipeline(arg_tys)
        cres_1 = test_pipeline.compile_extra(axy)
        nb_axy_1 = cres_1.entry_point

        control_pipeline2 = RewritesTester.mk_no_rw_pipeline(arg_tys)
        cres_2 = control_pipeline2.compile_extra(ax2)
        nb_ctl = cres_2.entry_point

        expected = nb_axy_0(A, X, Y)
        actual = nb_axy_1(A, X, Y)
        control = nb_ctl(A, X, Y)
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(control, actual)

        ir0 = control_pipeline.interp.blocks
        ir1 = test_pipeline.interp.blocks
        ir2 = control_pipeline2.interp.blocks
        self.assertEqual(len(ir0), len(ir1))
        self.assertEqual(len(ir0), len(ir2))
        # The rewritten IR should be smaller than the original.
        self.assertGreater(len(ir0[0].body), len(ir1[0].body))
        self.assertEqual(len(ir0[0].body), len(ir2[0].body))

    def _get_array_exprs(self, block):
        for instr in block:
            if isinstance(instr, ir.Assign):
                if isinstance(instr.value, ir.Expr):
                    if instr.value.op == 'arrayexpr':
                        yield instr

    def _array_expr_to_set(self, expr, out=None):
        '''
        Convert an array expression tree into a set of operators.
        '''
        if out is None:
            out = set()
        if not isinstance(expr, tuple):
            raise ValueError("{0} not a tuple".format(expr))
        operation, operands = expr
        processed_operands = []
        for operand in operands:
            if isinstance(operand, tuple):
                operand, _ = self._array_expr_to_set(operand, out)
            processed_operands.append(operand)
        processed_expr = operation, tuple(processed_operands)
        out.add(processed_expr)
        return processed_expr, out

    def _test_root_function(self, fn=pos_root):
        A = np.random.random(10)
        B = np.random.random(10) + 1. # Increase likelihood of real
                                      # root (could add 2 to force all
                                      # roots to be real).
        C = np.random.random(10)
        arg_tys = [typeof(arg) for arg in (A, B, C)]

        control_pipeline = RewritesTester.mk_no_rw_pipeline(arg_tys)
        control_cres = control_pipeline.compile_extra(fn)
        nb_fn_0 = control_cres.entry_point

        test_pipeline = RewritesTester.mk_pipeline(arg_tys)
        test_cres = test_pipeline.compile_extra(fn)
        nb_fn_1 = test_cres.entry_point

        np_result = fn(A, B, C)
        nb_result_0 = nb_fn_0(A, B, C)
        nb_result_1 = nb_fn_1(A, B, C)
        np.testing.assert_array_almost_equal(np_result, nb_result_0)
        np.testing.assert_array_almost_equal(nb_result_0, nb_result_1)

        return Namespace(locals())

    def _assert_total_rewrite(self, ir0, ir1):
        '''
        Given two dictionaries of Numba IR blocks, check to make sure the
        control IR (ir0) has no array expressions, while the test IR
        contains one and only one.
        '''
        self.assertEqual(len(ir0), len(ir1))
        self.assertGreater(len(ir0[0].body), len(ir1[0].body))
        self.assertEqual(len(list(self._get_array_exprs(ir0[0].body))), 0)
        self.assertEqual(len(list(self._get_array_exprs(ir1[0].body))), 1)

    def test_complicated_expr(self):
        '''
        Using the polynomial root function, ensure the full expression is
        being put in the same kernel with no remnants of intermediate
        array expressions.
        '''
        ns = self._test_root_function()
        self._assert_total_rewrite(ns.control_pipeline.interp.blocks,
                                   ns.test_pipeline.interp.blocks)

    def test_common_subexpressions(self, fn=neg_root_common_subexpr):
        '''
        Attempt to verify that rewriting will incorporate user common
        subexpressions properly.
        '''
        ns = self._test_root_function(fn)
        ir0 = ns.control_pipeline.interp.blocks
        ir1 = ns.test_pipeline.interp.blocks
        self.assertEqual(len(ir0), len(ir1))
        self.assertGreater(len(ir0[0].body), len(ir1[0].body))
        self.assertEqual(len(list(self._get_array_exprs(ir0[0].body))), 0)
        # Verify that we didn't rewrite everything into a monolithic
        # array expression since we stored temporary values in
        # variables that might be used later (from the optimization's
        # point of view).
        array_expr_instrs = list(self._get_array_exprs(ir1[0].body))
        self.assertGreater(len(array_expr_instrs), 1)
        # Now check that we haven't duplicated any subexpressions in
        # the rewritten code.
        array_sets = list(self._array_expr_to_set(instr.value.expr)[1]
                          for instr in array_expr_instrs)
        for expr_set_0, expr_set_1 in zip(array_sets[:-1], array_sets[1:]):
            intersections = expr_set_0.intersection(expr_set_1)
            if intersections:
                self.fail("Common subexpressions detected in array "
                          "expressions ({0})".format(intersections))

    def test_complex_subexpression(self):
        return self.test_common_subexpressions(neg_root_complex_subexpr)

    def test_ufunc_and_dufunc_calls(self):
        '''
        Verify that ufunc and DUFunc calls are being properly included in
        array expressions.
        '''
        A = np.random.random(10)
        B = np.random.random(10)
        arg_tys = [typeof(arg) for arg in (A, B)]

        vaxy_descr = vaxy._dispatcher.targetdescr
        control_pipeline = RewritesTester.mk_no_rw_pipeline(
            arg_tys,
            typing_context=vaxy_descr.typing_context,
            target_context=vaxy_descr.target_context)
        cres_0 = control_pipeline.compile_extra(call_stuff)
        nb_call_stuff_0 = cres_0.entry_point

        test_pipeline = RewritesTester.mk_pipeline(
            arg_tys,
            typing_context=vaxy_descr.typing_context,
            target_context=vaxy_descr.target_context)
        cres_1 = test_pipeline.compile_extra(call_stuff)
        nb_call_stuff_1 = cres_1.entry_point

        expected = call_stuff(A, B)
        control = nb_call_stuff_0(A, B)
        actual = nb_call_stuff_1(A, B)
        np.testing.assert_array_almost_equal(expected, control)
        np.testing.assert_array_almost_equal(expected, actual)

        self._assert_total_rewrite(control_pipeline.interp.blocks,
                                   test_pipeline.interp.blocks)

    def test_cmp_op(self):
        '''
        Verify that comparison operators are supported by the rewriter.
        '''
        ns = self._test_root_function(are_roots_imaginary)
        self._assert_total_rewrite(ns.control_pipeline.interp.blocks,
                                   ns.test_pipeline.interp.blocks)


class TestRewriteIssues(unittest.TestCase):
    def test_issue_1184(self):
        from numba import jit
        import numpy as np

        @jit(nopython=True)
        def foo(arr):
            return arr

        @jit(nopython=True)
        def bar(arr):
            c = foo(arr)
            d = foo(arr)   # two calls to trigger rewrite
            return c, d

        arr = np.arange(10)
        out_c, out_d = bar(arr)
        self.assertIs(out_c, out_d)
        self.assertIs(out_c, arr)



if __name__ == "__main__":
    unittest.main()
