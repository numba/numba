from __future__ import print_function, division, absolute_import
from timeit import Timer

import numpy as np

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
    _Bs2_4AsCs = (Bs ** 2. - _4AsCs) + 0j # Force into the complex domain.
    return (-Bs - (_Bs2_4AsCs ** 0.5)) / _2As


class RewritesTester(Pipeline):
    @classmethod
    def mk_pipeline(cls, args, return_type=None, flags=None, locals={},
                    library=None):
        if not flags:
            flags = Flags()
        flags.nrt = True
        tyctx = typing.Context()
        tgtctx = cpu.CPUContext(tyctx)
        return cls(tyctx, tgtctx, library, args, return_type, flags,
                   locals)

    @classmethod
    def mk_no_rw_pipeline(cls, args, return_type=None, flags=None, locals={},
                          library=None):
        if not flags:
            flags = Flags()
        flags.no_rewrites = True
        return cls.mk_pipeline(args, return_type, flags, locals, library)


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
        test_cres = test_pipeline.compile_extra(pos_root)
        nb_fn_1 = test_cres.entry_point

        np_result = fn(A, B, C)
        nb_result_0 = nb_fn_0(A, B, C)
        nb_result_1 = nb_fn_1(A, B, C)
        np.testing.assert_array_almost_equal(np_result, nb_result_0)
        np.testing.assert_array_almost_equal(nb_result_0, nb_result_1)

        return Namespace(locals())

    def test_complicated_expr(self):
        '''
        Using the polynomial root function, ensure the full expression is
        being put in the same kernel with no remnants of intermediate
        array expressions.
        '''
        ns = self._test_root_function()
        ir0 = ns.control_pipeline.interp.blocks
        ir1 = ns.test_pipeline.interp.blocks
        self.assertEqual(len(ir0), len(ir1))
        self.assertGreater(len(ir0[0].body), len(ir1[0].body))
        self.assertEqual(len(list(self._get_array_exprs(ir0[0].body))), 0)
        self.assertEqual(len(list(self._get_array_exprs(ir1[0].body))), 1)

    def test_common_subexpressions(self):
        '''
        Attempt to verify that rewriting will incorporate user common
        subexpressions properly.
        '''
        import pudb; pudb.set_trace()
        ns = self._test_root_function(neg_root_common_subexpr)


if __name__ == "__main__":
    unittest.main()
