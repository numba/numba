from __future__ import print_function

import sys

# import numpy in two ways, both uses needed
import numpy as np
import numpy

import numba.unittest_support as unittest
from numba import njit, jit, testing, utils
from numba.errors import NotDefinedError, TypingError, LoweringError
from .support import TestCase, tag


class TestClosure(TestCase):

    def run_jit_closure_variable(self, **jitargs):
        Y = 10

        def add_Y(x):
            return x + Y

        c_add_Y = jit('i4(i4)', **jitargs)(add_Y)
        self.assertEqual(c_add_Y(1), 11)

        # Like globals in Numba, the value of the closure is captured
        # at time of JIT
        Y = 12  # should not affect function
        self.assertEqual(c_add_Y(1), 11)

    def test_jit_closure_variable(self):
        self.run_jit_closure_variable(forceobj=True)

    def test_jit_closure_variable_npm(self):
        self.run_jit_closure_variable(nopython=True)

    def run_rejitting_closure(self, **jitargs):
        Y = 10

        def add_Y(x):
            return x + Y

        c_add_Y = jit('i4(i4)', **jitargs)(add_Y)
        self.assertEqual(c_add_Y(1), 11)

        # Redo the jit
        Y = 12
        c_add_Y_2 = jit('i4(i4)', **jitargs)(add_Y)
        self.assertEqual(c_add_Y_2(1), 13)
        Y = 13  # should not affect function
        self.assertEqual(c_add_Y_2(1), 13)

        self.assertEqual(c_add_Y(1), 11)  # Test first function again

    def test_rejitting_closure(self):
        self.run_rejitting_closure(forceobj=True)

    def test_rejitting_closure_npm(self):
        self.run_rejitting_closure(nopython=True)

    def run_jit_multiple_closure_variables(self, **jitargs):
        Y = 10
        Z = 2

        def add_Y_mult_Z(x):
            return (x + Y) * Z

        c_add_Y_mult_Z = jit('i4(i4)', **jitargs)(add_Y_mult_Z)
        self.assertEqual(c_add_Y_mult_Z(1), 22)

    def test_jit_multiple_closure_variables(self):
        self.run_jit_multiple_closure_variables(forceobj=True)

    def test_jit_multiple_closure_variables_npm(self):
        self.run_jit_multiple_closure_variables(nopython=True)

    def run_jit_inner_function(self, **jitargs):
        def mult_10(a):
            return a * 10

        c_mult_10 = jit('intp(intp)', **jitargs)(mult_10)
        c_mult_10.disable_compile()

        def do_math(x):
            return c_mult_10(x + 4)

        c_do_math = jit('intp(intp)', **jitargs)(do_math)
        c_do_math.disable_compile()

        with self.assertRefCount(c_do_math, c_mult_10):
            self.assertEqual(c_do_math(1), 50)

    def test_jit_inner_function(self):
        self.run_jit_inner_function(forceobj=True)

    def test_jit_inner_function_npm(self):
        self.run_jit_inner_function(nopython=True)

    @testing.allow_interpreter_mode
    def test_return_closure(self):

        def outer(x):

            def inner():
                return x + 1

            return inner

        cfunc = jit(outer)
        self.assertEqual(cfunc(10)(), outer(10)())


class TestInlinedClosure(TestCase):
    """
    Tests for (partial) closure support in njit. The support is partial
    because it only works for closures that can be successfully inlined
    at compile time.
    """

    @tag('important')
    def test_inner_function(self):

        def outer(x):

            def inner(x):
                return x * x

            return inner(x) + inner(x)

        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))

    @tag('important')
    def test_inner_function_with_closure(self):

        def outer(x):
            y = x + 1

            def inner(x):
                return x * x + y

            return inner(x) + inner(x)

        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))

    @tag('important')
    def test_inner_function_with_closure_2(self):

        def outer(x):
            y = x + 1

            def inner(x):
                return x * y

            y = inner(x)
            return y + inner(x)

        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))

    @unittest.skipIf(utils.PYVERSION < (3, 0), "needs Python 3")
    def test_inner_function_with_closure_3(self):

        code = """
            def outer(x):
                y = x + 1
                z = 0

                def inner(x):
                    nonlocal z
                    z += x * x
                    return z + y

                return inner(x) + inner(x) + z
        """
        ns = {}
        exec(code.strip(), ns)

        cfunc = njit(ns['outer'])
        self.assertEqual(cfunc(10), ns['outer'](10))

    @tag('important')
    def test_inner_function_nested(self):

        def outer(x):

            def inner(y):

                def innermost(z):
                    return x + y + z

                s = 0
                for i in range(y):
                    s += innermost(i)
                return s

            return inner(x * x)

        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))

    @tag('important')
    def test_bulk_use_cases(self):
        """ Tests the large number of use cases defined below """

        # jitted function used in some tests
        @njit
        def fib3(n):
            if n < 2:
                return n
            return fib3(n - 1) + fib3(n - 2)

        def outer1(x):
            """ Test calling recursive function from inner """
            def inner(x):
                return fib3(x)
            return inner(x)

        def outer2(x):
            """ Test calling recursive function from closure """
            z = x + 1

            def inner(x):
                return x + fib3(z)
            return inner(x)

        def outer3(x):
            """ Test recursive inner """
            def inner(x):
                if x + y < 2:
                    return 10
                else:
                    inner(x - 1)
            return inner(x)

        def outer4(x):
            """ Test recursive closure """
            y = x + 1

            def inner(x):
                if x + y < 2:
                    return 10
                else:
                    inner(x - 1)
            return inner(x)

        def outer5(x):
            """ Test nested closure """
            y = x + 1

            def inner1(x):
                z = y + x + 2

                def inner2(x):
                    return x + z

                return inner2(x) + y

            return inner1(x)

        def outer6(x):
            """ Test closure with list comprehension in body """
            y = x + 1

            def inner1(x):
                z = y + x + 2
                return [t for t in range(z)]
            return inner1(x)

        _OUTER_SCOPE_VAR = 9

        def outer7(x):
            """ Test use of outer scope var, no closure """
            z = x + 1
            return x + z + _OUTER_SCOPE_VAR

        _OUTER_SCOPE_VAR = 9

        def outer8(x):
            """ Test use of outer scope var, with closure """
            z = x + 1

            def inner(x):
                return x + z + _OUTER_SCOPE_VAR
            return inner(x)

        def outer9(x):
            """ Test closure assignment"""
            z = x + 1

            def inner(x):
                return x + z
            f = inner
            return f(x)

        def outer10(x):
            """ Test two inner, one calls other """
            z = x + 1

            def inner(x):
                return x + z

            def inner2(x):
                return inner(x)

            return inner2(x)

        def outer11(x):
            """ return the closure """
            z = x + 1

            def inner(x):
                return x + z
            return inner

        def outer12(x):
            """ closure with kwarg"""
            z = x + 1

            def inner(x, kw=7):
                return x + z + kw
            return inner(x)

        def outer13(x, kw=7):
            """ outer with kwarg no closure"""
            z = x + 1 + kw
            return z

        def outer14(x, kw=7):
            """ outer with kwarg used in closure"""
            z = x + 1

            def inner(x):
                return x + z + kw
            return inner(x)

        def outer15(x, kw=7):
            """ outer with kwarg as arg to closure"""
            z = x + 1

            def inner(x, kw):
                return x + z + kw
            return inner(x, kw)

        def outer16(x):
            """ closure is generator, consumed locally """
            z = x + 1

            def inner(x):
                yield x + z

            return list(inner(x))

        def outer17(x):
            """ closure is generator, returned """
            z = x + 1

            def inner(x):
                yield x + z

            return inner(x)

        def outer18(x):
            """ closure is generator, consumed in loop """
            z = x + 1

            def inner(x):
                yield x + z

            for i in inner(x):
                t = i

            return t

        def outer19(x):
            """ closure as arg to another closure """
            z1 = x + 1
            z2 = x + 2

            def inner(x):
                return x + z1

            def inner2(f, x):
                return f(x) + z2

            return inner2(inner, x)

        def outer20(x):
            #""" Test calling numpy in closure """
            z = x + 1

            def inner(x):
                return x + numpy.cos(z)
            return inner(x)

        def outer21(x):
            #""" Test calling numpy import as in closure """
            z = x + 1

            def inner(x):
                return x + np.cos(z)
            return inner(x)

        # functions to test that are expected to pass
        f = [outer1, outer2, outer5, outer6, outer7, outer8,
             outer9, outer10, outer12, outer13, outer14,
             outer15, outer19, outer20, outer21]
        for ref in f:
            cfunc = njit(ref)
            var = 10
            self.assertEqual(cfunc(var), ref(var))

        # test functions that are expected to fail
        with self.assertRaises(NotImplementedError) as raises:
            cfunc = jit(nopython=True)(outer3)
            cfunc(var)
        msg = "Unsupported use of op_LOAD_CLOSURE encountered"
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(NotImplementedError) as raises:
            cfunc = jit(nopython=True)(outer4)
            cfunc(var)
        msg = "Unsupported use of op_LOAD_CLOSURE encountered"
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            cfunc = jit(nopython=True)(outer11)
            cfunc(var)
        errcls = "type" if utils.PYVERSION < (3, 0) else "class"
        msg = "cannot determine Numba type of <" + errcls + " 'code'>"
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            cfunc = jit(nopython=True)(outer16)
            cfunc(var)
        msg = "with parameters (none)"
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(LoweringError) as raises:
            cfunc = jit(nopython=True)(outer17)
            cfunc(var)
        msg = "'NoneType' object has no attribute 'yield_points'"
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(TypingError) as raises:
            cfunc = jit(nopython=True)(outer18)
            cfunc(var)
        msg = "Invalid usage of getiter with parameters (none)"
        self.assertIn(msg, str(raises.exception))


if __name__ == '__main__':
    unittest.main()
