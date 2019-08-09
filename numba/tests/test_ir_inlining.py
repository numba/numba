"""
This tests the inline kwarg to @jit and @overload etc, it has nothing to do with
LLVM or low level inlining"
"""

from __future__ import print_function, absolute_import

from .support import TestCase, unittest
import numba
from numba import njit, ir
from numba.ir_utils import dead_code_elimination
from itertools import product, combinations


class InlineTestPipeline(numba.compiler.BasePipeline):
    """ Same as the standard pipeline, but preserves the func_ir into the
    metadata store"""

    def stage_preserve_final_ir(self):
        self.metadata['final_func_ir'] = self.func_ir.copy()

    def stage_dce(self):
        dead_code_elimination(self.func_ir, self.typemap)

    def define_pipelines(self, pm):
        self.define_nopython_pipeline(pm)
        # mangle the default pipeline and inject DCE and IR preservation ahead
        # of legalisation
        allstages = pm.pipeline_stages['nopython']
        new_pipe = []
        for x in allstages:
            if x[0] == self.stage_ir_legalization:
                new_pipe.append((self.stage_dce, "DCE"))
                new_pipe.append((self.stage_preserve_final_ir, "preserve IR"))
            new_pipe.append(x)
        pm.pipeline_stages['nopython'] = new_pipe


# this global has the same name as the the global in inlining_usecases.py, it
# is here to check that inlined functions bind to their own globals
_GLOBAL1 = -50


class TestFunctionInlining(TestCase):

    def check(self, test_impl, *args, inline_expect=None):
        assert inline_expect
        j_func = njit(pipeline_class=InlineTestPipeline)(test_impl)

        # check they produce the same answer first!
        self.assertEqual(test_impl(*args), j_func(*args))

        # make sure IR doesn't have branches
        fir = j_func.overloads[j_func.signatures[0]].metadata['final_func_ir']
        fir.blocks = numba.ir_utils.simplify_CFG(fir.blocks)
        fir.dump()
        self.assertEqual(len(fir.blocks), 1)
        block = next(iter(fir.blocks.values()))

        # if we don't expect the function to be inlined then make sure there is
        # 'call' present still
        exprs = [x for x in block.find_exprs()]
        assert exprs
        for k, v in inline_expect.items():
            found = False
            for expr in exprs:
                if getattr(expr, 'op', False) == 'call':
                    func_defn = fir.get_definition(expr.func)
                    found |= func_defn.name == k
            self.assertFalse(found == v)

    # check the options

    # check the cost model behaves

    # check that functions inline from:
    # * module level
    # * another module
    # * another module's submodule
    # * a factory

    def test_basic_inline_never(self):
        @njit(inline='never')
        def foo():
            return

        def impl():
            return foo()
        self.check(impl, inline_expect={'foo': False})

    def test_basic_inline_always(self):
        @njit(inline='always')
        def foo():
            return

        def impl():
            return foo()
        self.check(impl, inline_expect={'foo': True})

    def test_basic_inline_combos(self):

        def impl():
            x = foo()
            y = bar()
            z = baz()
            return x, y, z

        opts = (('always'), ('never'))

        for inline_foo, inline_bar, inline_baz in product(opts, opts, opts):

            @njit(inline=inline_foo)
            def foo():
                return

            @njit(inline=inline_bar)
            def bar():
                return

            @njit(inline=inline_baz)
            def baz():
                return

            inline_expect = {'foo': inline_foo, 'bar': inline_bar,
                             'baz': inline_baz}
            self.check(impl, inline_expect=inline_expect)

    @unittest.skip("Need to work out how to prevent this")
    def test_recursive_inline(self):

        @njit(inline='always')
        def foo(x):
            if x == 0:
                return 12
            else:
                foo(x - 1)

        a = 3

        def impl():
            b = 0
            if a > 1:
                b += 1
            foo(5)
            if b < a:
                b -= 1

        self.check(impl, inline_expect={'foo': True})

    def test_freevar_bindings(self):

        def factory(x, y):
            z = x + 12
            @njit
            def func():
                return (x, y + 3, z)
            return func

        def impl():
            x = foo()
            y = bar()
            z = baz()
            return x, y, z

        opts = (('always'), ('never'))

        for inline_foo, inline_bar, inline_baz in product(opts, opts, opts):

            foo = factory(10, 20)
            bar = factory(30, 40)
            baz = factory(50, 60)

            inline_expect = {'foo': inline_foo, 'bar': inline_bar,
                             'baz': inline_baz}
            self.check(impl, inline_expect=inline_expect)

    def test_inline_from_another_module(self):

        from .inlining_usecases import bar

        def impl():
            z = _GLOBAL1 + 2
            return bar(), z

        self.check(impl, inline_expect={'bar': True})

    def test_inline_w_freevar_from_another_module(self):

        from .inlining_usecases import baz_factory

        def gen(a, b):
            bar = baz_factory(a)

            def impl():
                z = _GLOBAL1 + a * b
                return bar(), z, a
            return impl

        impl = gen(10, 20)
        self.check(impl, inline_expect={'bar': True})

    def test_inlining_models(self):

        def sentinel_17_cost_model(caller_inline_info, callee_inline_info):
            # sentinel 17 cost model, this is a fake cost model that will return
            # True (i.e. inline) if the ir.Const(17) is found in the caller IR,
            # and the callee when executed returns 17. It's contrived and for
            # testing purposes only but is designed run both args.
            if callee_inline_info() == 17:
                for blk in caller_inline_info.blocks.values():
                    for stmt in blk.body:
                        if isinstance(stmt, ir.Assign):
                            if isinstance(stmt.value, ir.Const):
                                if stmt.value.value == 17:
                                    return True
            return False

        for ret in (10, 17):
            @njit(inline=sentinel_17_cost_model)
            def foo():
                return ret

            def impl(z):
                x = z + 17  # const 17 in caller
                y = foo()
                return y + 3, x

            self.check(impl, 10, inline_expect={'foo': ret == 17})
