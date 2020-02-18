"""
Tests for sub-components of parfors.
"""
import unittest

import numpy as np

import numba.parfors.parfor
from numba.core import (
    typing,
    rewrites,
    typed_passes,
    inline_closurecall,
    compiler,
    cpu,
)
from numba.core.registry import cpu_target
from numba.tests.support import TestCase


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


def run_parfor_sub_pass(test_func, args, **kws):
    # TODO: refactor this with get_optimized_numba_ir() where this is
    #       copied from
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

        inline_pass = inline_closurecall.InlineClosureCallPass(
            tp.state.func_ir, options, typed=True
        )
        inline_pass.run()

        rewrites.rewrite_registry.apply("before-inference", tp.state)

        (
            tp.state.typemap,
            tp.state.return_type,
            tp.state.calltypes,
        ) = typed_passes.type_inference_stage(
            tp.state.typingctx, tp.state.func_ir, tp.state.args, None
        )

        diagnostics = numba.parfors.parfor.ParforDiagnostics()

        preparfor_pass = numba.parfors.parfor.PreParforPass(
            tp.state.func_ir,
            tp.state.typemap,
            tp.state.calltypes,
            tp.state.typingctx,
            options,
            swapped=diagnostics.replaced_fns,
        )
        preparfor_pass.run()

        rewrites.rewrite_registry.apply("after-inference", tp.state)

        flags = compiler.Flags()
        parfor_pass = numba.parfors.parfor.ParforPass(
            tp.state.func_ir,
            tp.state.typemap,
            tp.state.calltypes,
            tp.state.return_type,
            tp.state.typingctx,
            options,
            flags,
            diagnostics=diagnostics,
        )
        parfor_pass._pre_run()
        # Run subpass
        sub_pass = numba.parfors.parfor.ConvertSetItemPass(parfor_pass)
        sub_pass.run(parfor_pass.func_ir.blocks)

    return sub_pass


class TestConvertSetItemPass(TestCase):
    def test_setitem_full_slice(self):
        def test_impl():
            n = 10
            a = np.ones(n)
            a[:] = 7
            return a

        sub_pass = run_parfor_sub_pass(test_impl, ())
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record["reason"], "slice")

    def test_setitem_slice_stop_bound(self):
        def test_impl():
            n = 10
            a = np.ones(n)
            a[:5] = 7
            return a

        sub_pass = run_parfor_sub_pass(test_impl, ())
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record["reason"], "slice")

    def test_setitem_slice_start_bound(self):
        def test_impl():
            n = 10
            a = np.ones(n)
            a[4:] = 7
            return a

        sub_pass = run_parfor_sub_pass(test_impl, ())
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record["reason"], "slice")

    def test_setitem_gather_if_scalar(self):
        def test_impl():
            n = 10
            a = np.ones(n)
            b = np.ones_like(a, dtype=np.bool_)
            a[b] = 7
            return a

        sub_pass = run_parfor_sub_pass(test_impl, ())
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record["reason"], "masked_assign_boardcast_scalar")

    def test_setitem_gather_if_array(self):
        def test_impl():
            n = 10
            a = np.ones(n)
            b = np.ones_like(a, dtype=np.bool_)
            c = np.ones_like(a)
            a[b] = c[b]
            return a

        sub_pass = run_parfor_sub_pass(test_impl, ())
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record["reason"], "masked_assign_array")


if __name__ == "__main__":
    unittest.main()
