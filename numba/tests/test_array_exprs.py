from __future__ import print_function, division, absolute_import
from timeit import Timer

import numpy as np

from numba import unittest_support as unittest
from numba import compiler, typing, typeof
from numba.compiler import Pipeline, _PipelineManager, Flags
from numba.targets import cpu


def axy(a, x, y):
    return a * x + y

def ax2(a, x, y):
    return a * x + y


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
        A = np.linspace(0,1,10)
        X = np.linspace(2,1,10)
        Y = np.linspace(1,2,10)
        arg_tys = [typeof(arg) for arg in A, X, Y]

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
        self.assertTrue(np.all(expected == actual))
        self.assertTrue(np.all(control == actual))

        ir0 = control_pipeline.interp.blocks
        ir1 = test_pipeline.interp.blocks
        ir2 = control_pipeline2.interp.blocks
        self.assertEquals(len(ir0), len(ir1))
        self.assertEquals(len(ir0), len(ir2))
        # The rewritten IR should be smaller than the original.
        self.assertGreater(len(ir0[0].body), len(ir1[0].body))
        self.assertEquals(len(ir0[0].body), len(ir2[0].body))


if __name__ == "__main__":
    unittest.main()
