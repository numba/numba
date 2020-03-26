"""
This tests the inline kwarg to @jit and @overload etc, it has nothing to do with
LLVM or low level inlining.
"""


import numpy as np

from numba import njit, typeof
from numba.core import types, ir, ir_utils
from numba.core.extending import (
    overload,
    overload_method,
    overload_attribute,
    register_model,
    typeof_impl,
    unbox,
    NativeValue,
    register_jitable,
)
from numba.core.datamodel.models import OpaqueModel
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import IRLegalization
from numba.core.untyped_passes import PreserveIR
from itertools import product
from numba.tests.support import (TestCase, unittest, skip_py38_or_later,
                                 MemoryLeakMixin)


class InlineTestPipeline(CompilerBase):
    """ Same as the standard pipeline, but preserves the func_ir into the
    metadata store"""

    def define_pipelines(self):
        pipeline = DefaultPassBuilder.define_nopython_pipeline(
            self.state, "inliner_custom_pipe")
        # mangle the default pipeline and inject DCE and IR preservation ahead
        # of legalisation

        # TODO: add a way to not do this! un-finalizing is not a good idea
        pipeline._finalized = False
        pipeline.add_pass_after(PreserveIR, IRLegalization)

        pipeline.finalize()
        return [pipeline]


# this global has the same name as the global in inlining_usecases.py, it
# is here to check that inlined functions bind to their own globals
_GLOBAL1 = -50


@njit(inline='always')
def _global_func(x):
    return x + 1


# to be overloaded
def _global_defn(x):
    return x + 1


@overload(_global_defn, inline='always')
def _global_overload(x):
    return _global_defn


class InliningBase(TestCase):

    _DEBUG = False

    inline_opt_as_bool = {'always': True, 'never': False}

    # --------------------------------------------------------------------------
    # Example cost model

    def sentinel_17_cost_model(self, func_ir):
        # sentinel 17 cost model, this is a fake cost model that will return
        # True (i.e. inline) if the ir.FreeVar(17) is found in the func_ir,
        for blk in func_ir.blocks.values():
            for stmt in blk.body:
                if isinstance(stmt, ir.Assign):
                    if isinstance(stmt.value, ir.FreeVar):
                        if stmt.value.value == 17:
                            return True
        return False

    # --------------------------------------------------------------------------

    def check(self, test_impl, *args, **kwargs):
        inline_expect = kwargs.pop('inline_expect', None)
        assert inline_expect
        block_count = kwargs.pop('block_count', 1)
        assert not kwargs
        for k, v in inline_expect.items():
            assert isinstance(k, str)
            assert isinstance(v, bool)

        j_func = njit(pipeline_class=InlineTestPipeline)(test_impl)

        # check they produce the same answer first!
        self.assertEqual(test_impl(*args), j_func(*args))

        # make sure IR doesn't have branches
        fir = j_func.overloads[j_func.signatures[0]].metadata['preserved_ir']
        fir.blocks = ir_utils.simplify_CFG(fir.blocks)
        if self._DEBUG:
            print("FIR".center(80, "-"))
            fir.dump()
        if block_count != 'SKIP':
            self.assertEqual(len(fir.blocks), block_count)
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

        return fir  # for use in further analysis


# used in _gen_involved
_GLOBAL = 1234


def _gen_involved():
    _FREEVAR = 0xCAFE

    def foo(a, b, c=12, d=1j, e=None):
        f = a + b
        a += _FREEVAR
        g = np.zeros(c, dtype=np.complex64)
        h = f + g
        i = 1j / d
        # For SSA, zero init, n and t
        n = 0
        t = 0
        if np.abs(i) > 0:
            k = h / i
            l = np.arange(1, c + 1)
            m = np.sqrt(l - g) + e * k
            if np.abs(m[0]) < 1:
                for o in range(a):
                    n += 0
                    if np.abs(n) < 3:
                        break
                n += m[2]
            p = g / l
            q = []
            for r in range(len(p)):
                q.append(p[r])
                if r > 4 + 1:
                    s = 123
                    t = 5
                    if s > 122 - c:
                        t += s
                t += q[0] + _GLOBAL

        return f + o + r + t + r + a + n

    return foo


class TestFunctionInlining(MemoryLeakMixin, InliningBase):

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

            inline_expect = {'foo': self.inline_opt_as_bool[inline_foo],
                             'bar': self.inline_opt_as_bool[inline_bar],
                             'baz': self.inline_opt_as_bool[inline_baz]}
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

        def factory(inline, x, y):
            z = x + 12
            @njit(inline=inline)
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

            foo = factory(inline_foo, 10, 20)
            bar = factory(inline_bar, 30, 40)
            baz = factory(inline_baz, 50, 60)

            inline_expect = {'foo': self.inline_opt_as_bool[inline_foo],
                             'bar': self.inline_opt_as_bool[inline_bar],
                             'baz': self.inline_opt_as_bool[inline_baz]}
            self.check(impl, inline_expect=inline_expect)

    def test_global_binding(self):

        def impl():
            x = 19
            return _global_func(x)

        self.check(impl, inline_expect={'_global_func': True})

    def test_inline_from_another_module(self):

        from .inlining_usecases import bar

        def impl():
            z = _GLOBAL1 + 2
            return bar(), z

        self.check(impl, inline_expect={'bar': True})

    def test_inline_from_another_module_w_getattr(self):

        import numba.tests.inlining_usecases as iuc

        def impl():
            z = _GLOBAL1 + 2
            return iuc.bar(), z

        self.check(impl, inline_expect={'bar': True})

    def test_inline_from_another_module_w_2_getattr(self):

        import numba.tests.inlining_usecases  # noqa forces registration
        import numba.tests as nt

        def impl():
            z = _GLOBAL1 + 2
            return nt.inlining_usecases.bar(), z

        self.check(impl, inline_expect={'bar': True})

    def test_inline_from_another_module_as_freevar(self):

        def factory():
            from .inlining_usecases import bar
            @njit(inline='always')
            def tmp():
                return bar()
            return tmp

        baz = factory()

        def impl():
            z = _GLOBAL1 + 2
            return baz(), z

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

        def s17_caller_model(expr, caller_info, callee_info):
            self.assertIsInstance(expr, ir.Expr)
            self.assertEqual(expr.op, "call")
            return self.sentinel_17_cost_model(caller_info)

        def s17_callee_model(expr, caller_info, callee_info):
            self.assertIsInstance(expr, ir.Expr)
            self.assertEqual(expr.op, "call")
            return self.sentinel_17_cost_model(callee_info)

        # caller has sentinel
        for caller, callee in ((11, 17), (17, 11)):

            @njit(inline=s17_caller_model)
            def foo():
                return callee

            def impl(z):
                x = z + caller
                y = foo()
                return y + 3, x

            self.check(impl, 10, inline_expect={'foo': caller == 17})

        # callee has sentinel
        for caller, callee in ((11, 17), (17, 11)):

            @njit(inline=s17_callee_model)
            def bar():
                return callee

            def impl(z):
                x = z + caller
                y = bar()
                return y + 3, x

            self.check(impl, 10, inline_expect={'bar': callee == 17})

    def test_inline_inside_loop(self):
        @njit(inline='always')
        def foo():
            return 12

        def impl():
            acc = 0.0
            for i in range(5):
                acc += foo()
            return acc

        self.check(impl, inline_expect={'foo': True}, block_count=4)

    def test_inline_inside_closure_inside_loop(self):
        @njit(inline='always')
        def foo():
            return 12

        def impl():
            acc = 0.0
            for i in range(5):
                def bar():
                    return foo() + 7
                acc += bar()
            return acc

        self.check(impl, inline_expect={'foo': True}, block_count=4)

    def test_inline_closure_inside_inlinable_inside_closure(self):
        @njit(inline='always')
        def foo(a):
            def baz():
                return 12 + a
            return baz() + 8

        def impl():
            z = 9

            def bar(x):
                return foo(z) + 7 + x
            return bar(z + 2)

        self.check(impl, inline_expect={'foo': True}, block_count=1)

    @skip_py38_or_later
    def test_inline_involved(self):

        fortran = njit(inline='always')(_gen_involved())

        @njit(inline='always')
        def boz(j):
            acc = 0

            def biz(t):
                return t + acc
            for x in range(j):
                acc += biz(8 + acc) + fortran(2., acc, 1, 12j, biz(acc))
            return acc

        @njit(inline='always')
        def foo(a):
            acc = 0
            for p in range(12):
                tmp = fortran(1, 1, 1, 1, 1)

                def baz(x):
                    return 12 + a + x + tmp
                acc += baz(p) + 8 + boz(p) + tmp
            return acc + baz(2)

        def impl():
            z = 9

            def bar(x):
                return foo(z) + 7 + x
            return bar(z + 2)

        self.check(impl, inline_expect={'foo': True, 'boz': True,
                                        'fortran': True}, block_count=37)


class TestRegisterJitableInlining(MemoryLeakMixin, InliningBase):

    def test_register_jitable_inlines(self):

        @register_jitable(inline='always')
        def foo():
            return 1

        def impl():
            foo()

        self.check(impl, inline_expect={'foo': True})


class TestOverloadInlining(MemoryLeakMixin, InliningBase):

    def test_basic_inline_never(self):
        def foo():
            pass

        @overload(foo, inline='never')
        def foo_overload():
            def foo_impl():
                pass
            return foo_impl

        def impl():
            return foo()

        self.check(impl, inline_expect={'foo': False})

    def test_basic_inline_always(self):

        def foo():
            pass

        @overload(foo, inline='always')
        def foo_overload():
            def impl():
                pass
            return impl

        def impl():
            return foo()

        self.check(impl, inline_expect={'foo': True})

    def test_inline_always_kw_no_default(self):
        # pass call arg by name that doesn't have default value
        def foo(a, b):
            return a + b

        @overload(foo, inline='always')
        def overload_foo(a, b):
            return lambda a, b: a + b

        def impl():
            return foo(3, b=4)

        self.check(impl, inline_expect={'foo': True})

    def test_inline_stararg_error(self):
        def foo(a, *b):
            return a + b[0]

        @overload(foo, inline='always')
        def overload_foo(a, *b):
            return lambda a, *b: a + b[0]

        def impl():
            return foo(3, 3, 5)

        with self.assertRaises(NotImplementedError) as e:
            self.check(impl, inline_expect={'foo': True})

        self.assertIn("Stararg not supported in inliner for arg 1 *b",
                      str(e.exception))

    def test_basic_inline_combos(self):

        def impl():
            x = foo()
            y = bar()
            z = baz()
            return x, y, z

        opts = (('always'), ('never'))

        for inline_foo, inline_bar, inline_baz in product(opts, opts, opts):

            def foo():
                pass

            def bar():
                pass

            def baz():
                pass

            @overload(foo, inline=inline_foo)
            def foo_overload():
                def impl():
                    return
                return impl

            @overload(bar, inline=inline_bar)
            def bar_overload():
                def impl():
                    return
                return impl

            @overload(baz, inline=inline_baz)
            def baz_overload():
                def impl():
                    return
                return impl

            inline_expect = {'foo': self.inline_opt_as_bool[inline_foo],
                             'bar': self.inline_opt_as_bool[inline_bar],
                             'baz': self.inline_opt_as_bool[inline_baz]}
            self.check(impl, inline_expect=inline_expect)

    def test_freevar_bindings(self):

        def impl():
            x = foo()
            y = bar()
            z = baz()
            return x, y, z

        opts = (('always'), ('never'))

        for inline_foo, inline_bar, inline_baz in product(opts, opts, opts):
            # need to repeatedly clobber definitions of foo, bar, baz so
            # @overload binds to the right instance WRT inlining

            def foo():
                x = 10
                y = 20
                z = x + 12
                return (x, y + 3, z)

            def bar():
                x = 30
                y = 40
                z = x + 12
                return (x, y + 3, z)

            def baz():
                x = 60
                y = 80
                z = x + 12
                return (x, y + 3, z)

            def factory(target, x, y, inline=None):
                z = x + 12
                @overload(target, inline=inline)
                def func():
                    def impl():
                        return (x, y + 3, z)
                    return impl

            factory(foo, 10, 20, inline=inline_foo)
            factory(bar, 30, 40, inline=inline_bar)
            factory(baz, 60, 80, inline=inline_baz)

            inline_expect = {'foo': self.inline_opt_as_bool[inline_foo],
                             'bar': self.inline_opt_as_bool[inline_bar],
                             'baz': self.inline_opt_as_bool[inline_baz]}

            self.check(impl, inline_expect=inline_expect)

    def test_global_overload_binding(self):

        def impl():
            z = 19
            return _global_defn(z)

        self.check(impl, inline_expect={'_global_defn': True})

    def test_inline_from_another_module(self):

        from .inlining_usecases import baz

        def impl():
            z = _GLOBAL1 + 2
            return baz(), z

        self.check(impl, inline_expect={'baz': True})

    def test_inline_from_another_module_w_getattr(self):

        import numba.tests.inlining_usecases as iuc

        def impl():
            z = _GLOBAL1 + 2
            return iuc.baz(), z

        self.check(impl, inline_expect={'baz': True})

    def test_inline_from_another_module_w_2_getattr(self):

        import numba.tests.inlining_usecases  # noqa forces registration
        import numba.tests as nt

        def impl():
            z = _GLOBAL1 + 2
            return nt.inlining_usecases.baz(), z

        self.check(impl, inline_expect={'baz': True})

    def test_inline_from_another_module_as_freevar(self):

        def factory():
            from .inlining_usecases import baz
            @njit(inline='always')
            def tmp():
                return baz()
            return tmp

        bop = factory()

        def impl():
            z = _GLOBAL1 + 2
            return bop(), z

        self.check(impl, inline_expect={'baz': True})

    def test_inline_w_freevar_from_another_module(self):

        from .inlining_usecases import bop_factory

        def gen(a, b):
            bar = bop_factory(a)

            def impl():
                z = _GLOBAL1 + a * b
                return bar(), z, a
            return impl

        impl = gen(10, 20)
        self.check(impl, inline_expect={'bar': True})

    def test_inlining_models(self):

        def s17_caller_model(expr, caller_info, callee_info):
            self.assertIsInstance(expr, ir.Expr)
            self.assertEqual(expr.op, "call")
            return self.sentinel_17_cost_model(caller_info.func_ir)

        def s17_callee_model(expr, caller_info, callee_info):
            self.assertIsInstance(expr, ir.Expr)
            self.assertEqual(expr.op, "call")
            return self.sentinel_17_cost_model(callee_info.func_ir)

        # caller has sentinel
        for caller, callee in ((10, 11), (17, 11)):

            def foo():
                return callee

            @overload(foo, inline=s17_caller_model)
            def foo_ol():
                def impl():
                    return callee
                return impl

            def impl(z):
                x = z + caller
                y = foo()
                return y + 3, x

            self.check(impl, 10, inline_expect={'foo': caller == 17})

        # callee has sentinel
        for caller, callee in ((11, 17), (11, 10)):

            def bar():
                return callee

            @overload(bar, inline=s17_callee_model)
            def bar_ol():
                def impl():
                    return callee
                return impl

            def impl(z):
                x = z + caller
                y = bar()
                return y + 3, x

            self.check(impl, 10, inline_expect={'bar': callee == 17})

    def test_multiple_overloads_with_different_inline_characteristics(self):
        # check that having different inlining options for different overloads
        # of the same function works ok

        # this is the Python equiv of the overloads below
        def bar(x):
            if isinstance(typeof(x), types.Float):
                return x + 1234
            else:
                return x + 1

        @overload(bar, inline='always')
        def bar_int_ol(x):
            if isinstance(x, types.Integer):
                def impl(x):
                    return x + 1
                return impl

        @overload(bar, inline='never')
        def bar_float_ol(x):
            if isinstance(x, types.Float):
                def impl(x):
                    return x + 1234
                return impl

        def always_inline_cost_model(*args):
            return True

        @overload(bar, inline=always_inline_cost_model)
        def bar_complex_ol(x):
            if isinstance(x, types.Complex):
                def impl(x):
                    return x + 1
                return impl

        def impl():
            a = bar(1)  # integer literal, should inline
            b = bar(2.3)  # float literal, should not inline
            # complex literal, should inline by virtue of cost model
            c = bar(3j)
            return a + b + c

        # there should still be a `bar` not inlined
        fir = self.check(impl, inline_expect={'bar': False}, block_count=1)

        # check there is one call left in the IR
        block = next(iter(fir.blocks.items()))[1]
        calls = [x for x in block.find_exprs(op='call')]
        self.assertTrue(len(calls) == 1)

        # check that the constant "1234" is not in the IR
        consts = [x.value for x in block.find_insts(ir.Assign)
                  if isinstance(getattr(x, 'value', None), ir.Const)]
        for val in consts:
            self.assertNotEqual(val.value, 1234)


class TestOverloadMethsAttrsInlining(InliningBase):
    def setUp(self):
        # Use test_id to makesure no collision is possible.
        test_id = self.id()
        DummyType = type('DummyTypeFor{}'.format(test_id), (types.Opaque,), {})

        dummy_type = DummyType("my_dummy")
        register_model(DummyType)(OpaqueModel)

        class Dummy(object):
            pass

        @typeof_impl.register(Dummy)
        def typeof_Dummy(val, c):
            return dummy_type

        @unbox(DummyType)
        def unbox_index(typ, obj, c):
            return NativeValue(c.context.get_dummy_value())

        self.Dummy = Dummy
        self.DummyType = DummyType

    def check_method(self, test_impl, args, expected, block_count,
                     expects_inlined=True):
        j_func = njit(pipeline_class=InlineTestPipeline)(test_impl)
        # check they produce the same answer first!
        self.assertEqual(j_func(*args), expected)

        # make sure IR doesn't have branches
        fir = j_func.overloads[j_func.signatures[0]].metadata['preserved_ir']
        fir.blocks = fir.blocks
        self.assertEqual(len(fir.blocks), block_count)
        if expects_inlined:
            # assert no calls
            for block in fir.blocks.values():
                calls = list(block.find_exprs('call'))
                self.assertFalse(calls)
        else:
            # assert has call
            allcalls = []
            for block in fir.blocks.values():
                allcalls += list(block.find_exprs('call'))
            self.assertTrue(allcalls)

    def check_getattr(self, test_impl, args, expected, block_count,
                      expects_inlined=True):
        j_func = njit(pipeline_class=InlineTestPipeline)(test_impl)
        # check they produce the same answer first!
        self.assertEqual(j_func(*args), expected)

        # make sure IR doesn't have branches
        fir = j_func.overloads[j_func.signatures[0]].metadata['preserved_ir']
        fir.blocks = fir.blocks
        self.assertEqual(len(fir.blocks), block_count)
        if expects_inlined:
            # assert no getattr
            for block in fir.blocks.values():
                getattrs = list(block.find_exprs('getattr'))
                self.assertFalse(getattrs)
        else:
            # assert has getattr
            allgetattrs = []
            for block in fir.blocks.values():
                allgetattrs += list(block.find_exprs('getattr'))
            self.assertTrue(allgetattrs)

    def test_overload_method_default_args_always(self):
        @overload_method(self.DummyType, "inline_method", inline='always')
        def _get_inlined_method(obj, val=None, val2=None):
            def get(obj, val=None, val2=None):
                return ("THIS IS INLINED", val, val2)
            return get

        def foo(obj):
            return obj.inline_method(123), obj.inline_method(val2=321)

        self.check_method(
            test_impl=foo,
            args=[self.Dummy()],
            expected=(("THIS IS INLINED", 123, None),
                      ("THIS IS INLINED", None, 321)),
            block_count=1,
        )

    def make_overload_method_test(self, costmodel, should_inline):
        def costmodel(*args):
            return should_inline

        @overload_method(self.DummyType, "inline_method", inline=costmodel)
        def _get_inlined_method(obj, val):
            def get(obj, val):
                return ("THIS IS INLINED!!!", val)
            return get

        def foo(obj):
            return obj.inline_method(123)

        self.check_method(
            test_impl=foo,
            args=[self.Dummy()],
            expected=("THIS IS INLINED!!!", 123),
            block_count=1,
            expects_inlined=should_inline,
        )

    def test_overload_method_cost_driven_always(self):
        self.make_overload_method_test(
            costmodel='always',
            should_inline=True,
        )

    def test_overload_method_cost_driven_never(self):
        self.make_overload_method_test(
            costmodel='never',
            should_inline=False,
        )

    def test_overload_method_cost_driven_must_inline(self):
        self.make_overload_method_test(
            costmodel=lambda *args: True,
            should_inline=True,
        )

    def test_overload_method_cost_driven_no_inline(self):
        self.make_overload_method_test(
            costmodel=lambda *args: False,
            should_inline=False,
        )

    def make_overload_attribute_test(self, costmodel, should_inline):
        @overload_attribute(self.DummyType, "inlineme", inline=costmodel)
        def _get_inlineme(obj):
            def get(obj):
                return "MY INLINED ATTRS"
            return get

        def foo(obj):
            return obj.inlineme

        self.check_getattr(
            test_impl=foo,
            args=[self.Dummy()],
            expected="MY INLINED ATTRS",
            block_count=1,
            expects_inlined=should_inline,
        )

    def test_overload_attribute_always(self):
        self.make_overload_attribute_test(
            costmodel='always',
            should_inline=True,
        )

    def test_overload_attribute_never(self):
        self.make_overload_attribute_test(
            costmodel='never',
            should_inline=False,
        )

    def test_overload_attribute_costmodel_must_inline(self):
        self.make_overload_attribute_test(
            costmodel=lambda *args: True,
            should_inline=True,
        )

    def test_overload_attribute_costmodel_no_inline(self):
        self.make_overload_attribute_test(
            costmodel=lambda *args: False,
            should_inline=False,
        )


class TestGeneralInlining(MemoryLeakMixin, InliningBase):

    def test_with_inlined_and_noninlined_variants(self):
        # This test is contrived and was to demonstrate fixing a bug in the
        # template walking logic where inlinable and non-inlinable definitions
        # would not mix.

        @overload(len, inline='always')
        def overload_len(A):
            if False:
                return lambda A: 10

        def impl():
            return len([2, 3, 4])

        # len(list) won't be inlined because the overload above doesn't apply
        self.check(impl, inline_expect={'len': False})

    def test_with_kwargs(self):

        def foo(a, b=3, c=5):
            return a + b + c

        @overload(foo, inline='always')
        def overload_foo(a, b=3, c=5):
            def impl(a, b=3, c=5):
                return a + b + c
            return impl

        def impl():
            return foo(3, c=10)

        self.check(impl, inline_expect={'foo': True})

    def test_with_kwargs2(self):

        @njit(inline='always')
        def bar(a, b=12, c=9):
            return a + b

        def impl(a, b=7, c=5):
            return bar(a + b, c=19)

        self.check(impl, 3, 4, inline_expect={'bar': True})

    def test_inlining_optional_constant(self):
        # This testcase causes `b` to be a Optional(bool) constant once it is
        # inlined into foo().
        @njit(inline='always')
        def bar(a=None, b=None):
            if b is None:
                b = 123     # this changes the type of `b` due to lack of SSA
            return (a, b)

        def impl():
            return bar(), bar(123), bar(b=321)

        self.check(impl, block_count='SKIP', inline_expect={'bar': True})


class TestInlineOptions(TestCase):

    def test_basic(self):
        always = InlineOptions('always')
        self.assertTrue(always.is_always_inline)
        self.assertFalse(always.is_never_inline)
        self.assertFalse(always.has_cost_model)
        self.assertEqual(always.value, 'always')

        never = InlineOptions('never')
        self.assertFalse(never.is_always_inline)
        self.assertTrue(never.is_never_inline)
        self.assertFalse(never.has_cost_model)
        self.assertEqual(never.value, 'never')

        def cost_model(x):
            return x
        model = InlineOptions(cost_model)
        self.assertFalse(model.is_always_inline)
        self.assertFalse(model.is_never_inline)
        self.assertTrue(model.has_cost_model)
        self.assertIs(model.value, cost_model)


if __name__ == '__main__':
    unittest.main()
