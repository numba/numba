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

class InlineTestPipeline(numba.compiler.BasePipeline):
    """compiler pipeline for testing inlining after optimization
    """
    def define_pipelines(self, pm):
        name = 'inline_test'
        pm.create_pipeline(name)
        self.add_preprocessing_stage(pm)
        self.add_with_handling_stage(pm)
        self.add_pre_typing_stage(pm)
        self.add_typing_stage(pm)
        pm.add_stage(self.stage_pre_parfor_pass, "Preprocessing for parfors")
        if not self.flags.no_rewrites:
            pm.add_stage(self.stage_nopython_rewrites, "nopython rewrites")
        if self.flags.auto_parallel.enabled:
            pm.add_stage(self.stage_parfor_pass, "convert to parfors")
        pm.add_stage(self.stage_inline_test_pass, "inline test")
        pm.add_stage(self.stage_ir_legalization,
                "ensure IR is legal prior to lowering")
        self.add_lowering_stage(pm)
        self.add_cleanup_stage(pm)
        pm.add_stage(self.stage_preserve_final_ir, "preserve IR")

    def stage_preserve_final_ir(self):
        self.metadata['final_func_ir'] = self.func_ir.copy()

    def stage_inline_test_pass(self):
        # assuming the function has one block with one call inside
        assert len(self.func_ir.blocks) == 1
        block = list(self.func_ir.blocks.values())[0]
        for i, stmt in enumerate(block.body):
            if guard(find_callname,self.func_ir, stmt.value) is not None:
                inline_closure_call(self.func_ir, {}, block, i, lambda: None,
                    self.typingctx, (), self.typemap, self.calltypes)
                break


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

        class InlineTestPipelinePrune(InlineTestPipeline):
            def stage_inline_test_pass(self):
                # assuming the function has one block with one call inside
                assert len(self.func_ir.blocks) == 1
                block = list(self.func_ir.blocks.values())[0]
                for i, stmt in enumerate(block.body):
                    if (guard(find_callname, self.func_ir, stmt.value)
                            is not None):
                        inline_closure_call(self.func_ir, {}, block, i,
                            foo.py_func, self.typingctx,
                            (self.typemap[stmt.value.args[0].name],),
                            self.typemap, self.calltypes)
                        break

        # make sure inline_closure_call runs in full pipeline
        j_func = njit(pipeline_class=InlineTestPipelinePrune)(test_impl)
        A = 3
        self.assertEqual(test_impl(A), j_func(A))
        self.assertEqual(test_impl(), j_func())

        # make sure IR doesn't have branches
        fir = j_func.overloads[(types.Omitted(None),)].metadata['final_func_ir']
        fir.blocks = numba.ir_utils.simplify_CFG(fir.blocks)
        self.assertEqual(len(fir.blocks), 1)

if __name__ == '__main__':
    unittest.main()
