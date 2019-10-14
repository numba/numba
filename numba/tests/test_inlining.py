from __future__ import print_function, absolute_import

import re
import numpy as np

from .support import TestCase, override_config, captured_stdout
import numba
from numba import unittest_support as unittest
from numba import jit, njit, types, ir, compiler
from numba.ir_utils import guard, find_callname, find_const, get_definition
from numba.targets.registry import CPUDispatcher
from numba.inline_closurecall import inline_closure_call
from .test_parfors import skip_unsupported

from numba.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
                             IRProcessing, DeadBranchPrune,
                             RewriteSemanticConstants, GenericRewrites,
                             WithLifting, PreserveIR, InlineClosureLikes)

from numba.typed_passes import (NopythonTypeInference, AnnotateTypes,
                           NopythonRewrites, PreParforPass, ParforPass,
                           DumpParforDiagnostics, NativeLowering,
                           IRLegalization, NoPythonBackend)

from numba.compiler_machinery import FunctionPass, PassManager, register_pass

@jit((types.int32,), nopython=True)
def inner(a):
    return a + 1

@jit((types.int32,), nopython=True)
def more(a):
    return inner(inner(a))

def outer_simple(a):
    return inner(a) * 2

def outer_multiple(a):
    return inner(a) * more(a)

@njit
def __dummy__():
    return

@register_pass(analysis_only=False, mutates_CFG=True)
class InlineTestPass(FunctionPass):
    _name = "inline_test_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        # assuming the function has one block with one call inside
        assert len(state.func_ir.blocks) == 1
        block = list(state.func_ir.blocks.values())[0]
        for i, stmt in enumerate(block.body):
            if guard(find_callname,state.func_ir, stmt.value) is not None:
                inline_closure_call(state.func_ir, {}, block, i, lambda: None,
                    state.typingctx, (), state.type_annotation.typemap,
                    state.type_annotation.calltypes)
                # also fix up the IR so that ir.Dels appear correctly/in correct
                # locations
                post_proc = numba.postproc.PostProcessor(state.func_ir)
                post_proc.run()
                break
        return True


def gen_pipeline(state, test_pass):
        name = 'inline_test'
        pm = PassManager(name)
        pm.add_pass(TranslateByteCode, "analyzing bytecode")
        pm.add_pass(FixupArgs, "fix up args")
        pm.add_pass(IRProcessing, "processing IR")
        pm.add_pass(WithLifting, "Handle with contexts")
        # pre typing
        if not state.flags.no_rewrites:
            pm.add_pass(GenericRewrites, "nopython rewrites")
            pm.add_pass(RewriteSemanticConstants, "rewrite semantic constants")
            pm.add_pass(DeadBranchPrune, "dead branch pruning")
        pm.add_pass(InlineClosureLikes,
                    "inline calls to locally defined closures")
        # typing
        pm.add_pass(NopythonTypeInference, "nopython frontend")
        pm.add_pass(AnnotateTypes, "annotate types")

        if state.flags.auto_parallel.enabled:
            pm.add_pass(PreParforPass, "Preprocessing for parfors")
        if not state.flags.no_rewrites:
            pm.add_pass(NopythonRewrites, "nopython rewrites")
        if state.flags.auto_parallel.enabled:
            pm.add_pass(ParforPass, "convert to parfors")

        pm.add_pass(test_pass, "inline test")

        # legalise
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")

        pm.add_pass(PreserveIR, "preserve IR")

        # lower
        pm.add_pass(NoPythonBackend, "nopython mode backend")
        pm.add_pass(DumpParforDiagnostics, "dump parfor diagnostics")
        return pm

class InlineTestPipeline(numba.compiler.CompilerBase):
    """compiler pipeline for testing inlining after optimization
    """
    def define_pipelines(self):
        pm = gen_pipeline(self.state, InlineTestPass)
        pm.finalize()
        return [pm]

class TestInlining(TestCase):
    """
    Check that jitted inner functions are inlined into outer functions,
    in nopython mode.
    Note that not all inner functions are guaranteed to be inlined.
    We just trust LLVM's inlining heuristics.
    """

    def make_pattern(self, fullname):
        """
        Make regexpr to match mangled name
        """
        parts = fullname.split('.')
        return r'_ZN?' + r''.join([r'\d+{}'.format(p) for p in parts])

    def assert_has_pattern(self, fullname, text):
        pat = self.make_pattern(fullname)
        self.assertIsNotNone(re.search(pat, text),
                             msg='expected {}'.format(pat))

    def assert_not_has_pattern(self, fullname, text):
        pat = self.make_pattern(fullname)
        self.assertIsNone(re.search(pat, text),
                          msg='unexpected {}'.format(pat))

    def test_inner_function(self):
        with override_config('DUMP_ASSEMBLY', True):
            with captured_stdout() as out:
                cfunc = jit((types.int32,), nopython=True)(outer_simple)
        self.assertPreciseEqual(cfunc(1), 4)
        # Check the inner function was elided from the output (which also
        # guarantees it was inlined into the outer function).
        asm = out.getvalue()
        prefix = __name__
        self.assert_has_pattern('%s.outer_simple' % prefix, asm)
        self.assert_not_has_pattern('%s.inner' % prefix, asm)

    def test_multiple_inner_functions(self):
        # Same with multiple inner functions, and multiple calls to
        # the same inner function (inner()).  This checks that linking in
        # the same library/module twice doesn't produce linker errors.
        with override_config('DUMP_ASSEMBLY', True):
            with captured_stdout() as out:
                cfunc = jit((types.int32,), nopython=True)(outer_multiple)
        self.assertPreciseEqual(cfunc(1), 6)
        asm = out.getvalue()
        prefix = __name__
        self.assert_has_pattern('%s.outer_multiple' % prefix, asm)
        self.assert_not_has_pattern('%s.more' % prefix, asm)
        self.assert_not_has_pattern('%s.inner' % prefix, asm)

    @skip_unsupported
    def test_inline_call_after_parfor(self):
        # replace the call to make sure inlining doesn't cause label conflict
        # with parfor body
        def test_impl(A):
            __dummy__()
            return A.sum()
        j_func = njit(parallel=True, pipeline_class=InlineTestPipeline)(
                                                                    test_impl)
        A = np.arange(10)
        self.assertEqual(test_impl(A), j_func(A))

    @skip_unsupported
    def test_inline_update_target_def(self):

        def test_impl(a):
            if a == 1:
                b = 2
            else:
                b = 3
            return b

        func_ir = compiler.run_frontend(test_impl)
        blocks = list(func_ir.blocks.values())
        for block in blocks:
            for i, stmt in enumerate(block.body):
                # match b = 2 and replace with lambda: 2
                if (isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var)
                        and guard(find_const, func_ir, stmt.value) == 2):
                    # replace expr with a dummy call
                    func_ir._definitions[stmt.target.name].remove(stmt.value)
                    stmt.value = ir.Expr.call(ir.Var(block.scope, "myvar", loc=stmt.loc), (), (), stmt.loc)
                    func_ir._definitions[stmt.target.name].append(stmt.value)
                    #func = g.py_func#
                    inline_closure_call(func_ir, {}, block, i, lambda: 2)
                    break

        self.assertEqual(len(func_ir._definitions['b']), 2)

    @skip_unsupported
    def test_inline_var_dict_ret(self):
        # make sure inline_closure_call returns the variable replacement dict
        # and it contains the original variable name used in locals
        @numba.njit(locals={'b': numba.float64})
        def g(a):
            b = a + 1
            return b

        def test_impl():
            return g(1)

        func_ir = compiler.run_frontend(test_impl)
        blocks = list(func_ir.blocks.values())
        for block in blocks:
            for i, stmt in enumerate(block.body):
                if (isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op == 'call'):
                    func_def = guard(get_definition, func_ir, stmt.value.func)
                    if (isinstance(func_def, (ir.Global, ir.FreeVar))
                            and isinstance(func_def.value, CPUDispatcher)):
                        py_func = func_def.value.py_func
                        _, var_map = inline_closure_call(
                            func_ir, py_func.__globals__, block, i, py_func)
                        break

        self.assertTrue('b' in var_map)

    @skip_unsupported
    def test_inline_call_branch_pruning(self):
        # branch pruning pass should run properly in inlining to enable
        # functions with type checks
        @njit
        def foo(A=None):
            if A is None:
                return 2
            else:
                return A

        def test_impl(A=None):
            return foo(A)

        @register_pass(analysis_only=False, mutates_CFG=True)
        class PruningInlineTestPass(FunctionPass):
            _name = "pruning_inline_test_pass"

            def __init__(self):
                FunctionPass.__init__(self)

            def run_pass(self, state):
                # assuming the function has one block with one call inside
                assert len(state.func_ir.blocks) == 1
                block = list(state.func_ir.blocks.values())[0]
                for i, stmt in enumerate(block.body):
                    if (guard(find_callname, state.func_ir, stmt.value)
                            is not None):
                        inline_closure_call(state.func_ir, {}, block, i,
                            foo.py_func, state.typingctx,
                            (state.type_annotation.typemap[stmt.value.args[0].name],),
                            state.type_annotation.typemap, state.calltypes)
                        break
                return True

        class InlineTestPipelinePrune(numba.compiler.CompilerBase):

            def define_pipelines(self):
                pm = gen_pipeline(self.state, PruningInlineTestPass)
                pm.finalize()
                return [pm]

        # make sure inline_closure_call runs in full pipeline
        j_func = njit(pipeline_class=InlineTestPipelinePrune)(test_impl)
        A = 3
        self.assertEqual(test_impl(A), j_func(A))
        self.assertEqual(test_impl(), j_func())

        # make sure IR doesn't have branches
        fir = j_func.overloads[(types.Omitted(None),)].metadata['preserved_ir']
        fir.blocks = numba.ir_utils.simplify_CFG(fir.blocks)
        self.assertEqual(len(fir.blocks), 1)

if __name__ == '__main__':
    unittest.main()
