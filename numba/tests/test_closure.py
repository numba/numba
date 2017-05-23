from __future__ import print_function

import sys

import numba.unittest_support as unittest
from numba import njit, jit, testing
from .support import TestCase


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

    def test_inner_function(self):

        def outer(x):

            def inner(x):
                return x * x

            return inner(x) + inner(x)

        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))

    def test_inner_function_with_closure(self):

        def outer(x):
            y = x + 1

            def inner(x):
                return x * x + y

            return inner(x) + inner(x)

        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))

    def test_inner_function_with_closure_2(self):

        def outer(x):
            y = x + 1

            def inner(x):
                return x * y

            y = inner(x)
            return y + inner(x)

        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))


    def test_inner_function_with_closure_3(self):

        def outer(x):
            y = x + 1
            z = 0

            def inner(x):
                nonlocal z
                z += x * x
                return z + y

            return inner(x) + inner(x) + z

        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))

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


if __name__ == '__main__':
    unittest.main()
