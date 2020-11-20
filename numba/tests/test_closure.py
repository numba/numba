# import numpy in two ways, both uses needed
import numpy as np
import numpy

import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.tests.support import TestCase


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
                if x < 2:
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
        msg = "Cannot capture the non-constant value"
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(UnsupportedError) as raises:
            cfunc = jit(nopython=True)(outer16)
            cfunc(var)
        msg = "The use of yield in a closure is unsupported."
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(UnsupportedError) as raises:
            cfunc = jit(nopython=True)(outer17)
            cfunc(var)
        msg = "The use of yield in a closure is unsupported."
        self.assertIn(msg, str(raises.exception))

        with self.assertRaises(UnsupportedError) as raises:
            cfunc = jit(nopython=True)(outer18)
            cfunc(var)
        msg = "The use of yield in a closure is unsupported."
        self.assertIn(msg, str(raises.exception))


class TestObjmodeFallback(TestCase):
    # These are all based on tests from real life issues where, predominantly,
    # the object mode fallback compilation path would fail as a result of the IR
    # being mutated by closure inlining in npm. Tests are named after issues,
    # all of which failed to compile as of 0.44.

    decorators = [jit, jit(forceobj=True)]

    def test_issue2955(self):

        def numbaFailure(scores, cooc):
            rows, cols = scores.shape
            for i in range(rows):
                coxv = scores[i]
                groups = sorted(set(coxv), reverse=True)
                [set(np.argwhere(coxv == x).flatten()) for x in groups]

        x = np.random.random((10, 10))
        y = np.abs((np.random.randn(10, 10) * 1.732)).astype(np.int)
        for d in self.decorators:
            d(numbaFailure)(x, y)

    def test_issue3239(self):

        def fit(X, y):
            if type(X) is not np.ndarray:
                X = np.array(X)

            if type(y) is not np.ndarray:
                y = np.array(y)

            m, _ = X.shape
            X = np.hstack((
                np.array([[1] for _ in range(m)]),
                X
            ))

            res = np.dot(np.dot(X, X.T), y)
            intercept = res[0]
            coefs = res[1:]
            return intercept, coefs

        for d in self.decorators:
            res = d(fit)(np.arange(10).reshape(1, 10),
                         np.arange(10).reshape(1, 10))
            exp = fit(np.arange(10).reshape(1, 10),
                      np.arange(10).reshape(1, 10))
            np.testing.assert_equal(res, exp)

    def test_issue3289(self):
        b = [(5, 124), (52, 5)]

        def a():
            [b[index] for index in [0, 1]]
            for x in range(5):
                pass
        for d in self.decorators:
            d(a)()

    def test_issue3413(self):

        def foo(data):
            # commenting out this line prevents the crash:
            t = max([len(m) for m in data['y']])

            mask = data['x'] == 0
            if any(mask):
                z = 15
            return t, z

        data = {'x': np.arange(5), 'y': [[1], [2, 3]]}
        for d in self.decorators:
            res = d(foo)(data)
            np.testing.assert_allclose(res, foo(data))

    def test_issue3659(self):

        def main():
            a = np.array(((1, 2), (3, 4)))
            return np.array([x for x in a])
        for d in self.decorators:
            res = d(main)()
            np.testing.assert_allclose(res, main())

    def test_issue3803(self):

        def center(X):
            np.array([np.float_(x) for x in X.T])
            np.array([np.float_(1) for _ in X.T])
            return X

        X = np.zeros((10,))
        for d in self.decorators:
            res = d(center)(X)
            np.testing.assert_allclose(res, center(X))


if __name__ == '__main__':
    unittest.main()
