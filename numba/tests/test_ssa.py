"""
Tests for SSA reconstruction
"""
import sys
import copy
import logging

import numpy as np

from numba import njit, jit, types
from numba.core import errors
from numba.extending import overload
from numba.tests.support import TestCase, override_config


_DEBUG = False

if _DEBUG:
    # Enable debug logger on SSA reconstruction
    ssa_logger = logging.getLogger("numba.core.ssa")
    ssa_logger.setLevel(level=logging.DEBUG)
    ssa_logger.addHandler(logging.StreamHandler(sys.stderr))


class SSABaseTest(TestCase):

    def check_func(self, func, *args):
        got = func(*copy.deepcopy(args))
        exp = func.py_func(*copy.deepcopy(args))
        self.assertEqual(got, exp)


class TestSSA(SSABaseTest):
    """
    Contains tests to help isolate problems in SSA
    """

    def test_argument_name_reused(self):
        @njit
        def foo(x):
            x += 1
            return x

        self.check_func(foo, 123)

    def test_if_else_redefine(self):
        @njit
        def foo(x, y):
            z = x * y
            if x < y:
                z = x
            else:
                z = y
            return z

        self.check_func(foo, 3, 2)
        self.check_func(foo, 2, 3)

    def test_sum_loop(self):
        @njit
        def foo(n):
            c = 0
            for i in range(n):
                c += i
            return c

        self.check_func(foo, 0)
        self.check_func(foo, 10)

    def test_sum_loop_2vars(self):
        @njit
        def foo(n):
            c = 0
            d = n
            for i in range(n):
                c += i
                d += n
            return c, d

        self.check_func(foo, 0)
        self.check_func(foo, 10)

    def test_sum_2d_loop(self):
        @njit
        def foo(n):
            c = 0
            for i in range(n):
                for j in range(n):
                    c += j
                c += i
            return c

        self.check_func(foo, 0)
        self.check_func(foo, 10)

    def check_undefined_var(self, should_warn):
        @njit
        def foo(n):
            if n:
                if n > 0:
                    c = 0
                return c
            else:
                # variable c is not defined in this branch
                c += 1
                return c

        if should_warn:
            with self.assertWarns(errors.NumbaWarning) as warns:
                # n=1 so we won't actually run the branch with the uninitialized
                self.check_func(foo, 1)
            self.assertIn("Detected uninitialized variable c",
                          str(warns.warning))
        else:
            self.check_func(foo, 1)

        with self.assertRaises(UnboundLocalError):
            foo.py_func(0)

    def test_undefined_var(self):
        with override_config('ALWAYS_WARN_UNINIT_VAR', 0):
            self.check_undefined_var(should_warn=False)
        with override_config('ALWAYS_WARN_UNINIT_VAR', 1):
            self.check_undefined_var(should_warn=True)

    def test_phi_propagation(self):
        @njit
        def foo(actions):
            n = 1

            i = 0
            ct = 0
            while n > 0 and i < len(actions):
                n -= 1

                while actions[i]:
                    if actions[i]:
                        if actions[i]:
                            n += 10
                        actions[i] -= 1
                    else:
                        if actions[i]:
                            n += 20
                        actions[i] += 1

                    ct += n
                ct += n
            return ct, n

        self.check_func(foo, np.array([1, 2]))

    def test_unhandled_undefined(self):
        def function1(arg1, arg2, arg3, arg4, arg5):
            # This function is auto-generated.
            if arg1:
                var1 = arg2
                var2 = arg3
                var3 = var2
                var4 = arg1
                return
            else:
                if arg2:
                    if arg4:
                        var5 = arg4         # noqa: F841
                        return
                    else:
                        var6 = var4
                        return
                    return var6
                else:
                    if arg5:
                        if var1:
                            if arg5:
                                var1 = var6
                                return
                            else:
                                var7 = arg2     # noqa: F841
                                return arg2
                            return
                        else:
                            if var2:
                                arg5 = arg2
                                return arg1
                            else:
                                var6 = var3
                                return var4
                            return
                        return
                    else:
                        var8 = var1
                        return
                    return var8
                var9 = var3         # noqa: F841
                var10 = arg5        # noqa: F841
                return var1

        # The argument values is not critical for re-creating the bug
        # because the bug is in compile-time.
        expect = function1(2, 3, 6, 0, 7)
        got = njit(function1)(2, 3, 6, 0, 7)
        self.assertEqual(expect, got)


class TestReportedSSAIssues(SSABaseTest):
    # Tests from issues
    # https://github.com/numba/numba/issues?q=is%3Aopen+is%3Aissue+label%3ASSA

    def test_issue2194(self):

        @njit
        def foo():
            V = np.empty(1)
            s = np.uint32(1)

            for i in range(s):
                V[i] = 1
            for i in range(s, 1):
                pass

        self.check_func(foo, )

    def test_issue3094(self):

        @njit
        def doit(x):
            return x

        @njit
        def foo(pred):
            if pred:
                x = True
            else:
                x = False
            # do something with x
            return doit(x)

        self.check_func(foo, False)

    def test_issue3931(self):

        @njit
        def foo(arr):
            for i in range(1):
                arr = arr.reshape(3 * 2)
                arr = arr.reshape(3, 2)
            return(arr)

        np.testing.assert_allclose(foo(np.zeros((3, 2))),
                                   foo.py_func(np.zeros((3, 2))))

    def test_issue3976(self):

        def overload_this(a):
            return 'dummy'

        @njit
        def foo(a):
            if a:
                s = 5
                s = overload_this(s)
            else:
                s = 'b'

            return s

        @overload(overload_this)
        def ol(a):
            return overload_this

        self.check_func(foo, True)

    def test_issue3979(self):

        @njit
        def foo(A, B):
            x = A[0]
            y = B[0]
            for i in A:
                x = i
            for i in B:
                y = i
            return x, y

        self.check_func(foo, (1, 2), ('A', 'B'))

    def test_issue5219(self):

        def overload_this(a, b=None):
            if isinstance(b, tuple):
                b = b[0]
            return b

        @overload(overload_this)
        def ol(a, b=None):
            b_is_tuple = isinstance(b, (types.Tuple, types.UniTuple))

            def impl(a, b=None):
                if b_is_tuple is True:
                    b = b[0]
                return b
            return impl

        @njit
        def test_tuple(a, b):
            overload_this(a, b)

        self.check_func(test_tuple, 1, (2, ))

    def test_issue5223(self):

        @njit
        def bar(x):
            if len(x) == 5:
                return x
            x = x.copy()
            for i in range(len(x)):
                x[i] += 1
            return x

        a = np.ones(5)
        a.flags.writeable = False

        np.testing.assert_allclose(bar(a), bar.py_func(a))

    def test_issue5243(self):

        @njit
        def foo(q):
            lin = np.array((0.1, 0.6, 0.3))
            stencil = np.zeros((3, 3))
            stencil[0, 0] = q[0, 0]
            return lin[0]

        self.check_func(foo, np.zeros((2, 2)))

    def test_issue5482_missing_variable_init(self):
        # Test error that lowering fails because variable is missing
        # a definition before use.
        @njit("(intp, intp, intp)")
        def foo(x, v, n):
            for i in range(n):
                if i == 0:
                    if i == x:
                        pass
                    else:
                        problematic = v
                else:
                    if i == x:
                        pass
                    else:
                        problematic = problematic + v
            return problematic

    def test_issue5482_objmode_expr_null_lowering(self):
        # Existing pipelines will not have the Expr.null in objmode.
        # We have to create a custom pipeline to force a SSA reconstruction
        # and stripping.
        from numba.core.compiler import CompilerBase, DefaultPassBuilder
        from numba.untyped_passes import ReconstructSSA, IRProcessing
        from numba.typed_passes import PreLowerStripPhis

        class CustomPipeline(CompilerBase):
            def define_pipelines(self):
                pm = DefaultPassBuilder.define_objectmode_pipeline(self.state)
                # Force SSA reconstruction and stripping
                pm.add_pass_after(ReconstructSSA, IRProcessing)
                pm.add_pass_after(PreLowerStripPhis, ReconstructSSA)
                pm.finalize()
                return [pm]

        @jit("(intp, intp, intp)", looplift=False,
             pipeline_class=CustomPipeline)
        def foo(x, v, n):
            for i in range(n):
                if i == n:
                    if i == x:
                        pass
                    else:
                        problematic = v
                else:
                    if i == x:
                        pass
                    else:
                        problematic = problematic + v
            return problematic

    def test_issue5493_unneeded_phi(self):
        # Test error that unneeded phi is inserted because variable does not
        # have a dominance definition.
        data = (np.ones(2), np.ones(2))
        A = np.ones(1)
        B = np.ones((1,1))

        def foo(m, n, data):
            if len(data) == 1:
                v0 = data[0]
            else:
                v0 = data[0]
                # Unneeded PHI node for `problematic` would be placed here
                for _ in range(1, len(data)):
                    v0 += A

            for t in range(1, m):
                for idx in range(n):
                    t = B

                    if idx == 0:
                        if idx == n - 1:
                            pass
                        else:
                            problematic = t
                    else:
                        if idx == n - 1:
                            pass
                        else:
                            problematic = problematic + t
            return problematic

        expect = foo(10, 10, data)
        res1 = njit(foo)(10, 10, data)
        res2 = jit(forceobj=True, looplift=False)(foo)(10, 10, data)
        np.testing.assert_array_equal(expect, res1)
        np.testing.assert_array_equal(expect, res2)

    def test_issue5623_equal_statements_in_same_bb(self):

        def foo(pred, stack):
            i = 0
            c = 1

            if pred is True:
                stack[i] = c
                i += 1
                stack[i] = c
                i += 1

        python = np.array([0, 666])
        foo(True, python)

        nb = np.array([0, 666])
        njit(foo)(True, nb)

        expect = np.array([1, 1])

        np.testing.assert_array_equal(python, expect)
        np.testing.assert_array_equal(nb, expect)
