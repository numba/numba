"""
Testing object mode specifics.

"""

import numpy as np

import unittest
from numba.core.compiler import compile_isolated, Flags
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase


def complex_constant(n):
    tmp = n + 4
    return tmp + 3j


def long_constant(n):
    return n + 100000000000000000000000000000000000000000000000


def delitem_usecase(x):
    del x[:]


forceobj = Flags()
forceobj.force_pyobject = True


def loop_nest_3(x, y):
    n = 0
    for i in range(x):
        for j in range(y):
            for k in range(x + y):
                n += i * j

    return n


def array_of_object(x):
    return x


class TestObjectMode(TestCase):

    def test_complex_constant(self):
        pyfunc = complex_constant
        cres = compile_isolated(pyfunc, (), flags=forceobj)
        cfunc = cres.entry_point
        self.assertPreciseEqual(pyfunc(12), cfunc(12))

    def test_long_constant(self):
        pyfunc = long_constant
        cres = compile_isolated(pyfunc, (), flags=forceobj)
        cfunc = cres.entry_point
        self.assertPreciseEqual(pyfunc(12), cfunc(12))

    def test_loop_nest(self):
        """
        Test bug that decref the iterator early.
        If the bug occurs, a segfault should occur
        """
        pyfunc = loop_nest_3
        cres = compile_isolated(pyfunc, (), flags=forceobj)
        cfunc = cres.entry_point
        self.assertEqual(pyfunc(5, 5), cfunc(5, 5))

        def bm_pyfunc():
            pyfunc(5, 5)

        def bm_cfunc():
            cfunc(5, 5)

        print(utils.benchmark(bm_pyfunc))
        print(utils.benchmark(bm_cfunc))

    def test_array_of_object(self):
        cfunc = jit(array_of_object)
        objarr = np.array([object()] * 10)
        self.assertIs(cfunc(objarr), objarr)

    def test_sequence_contains(self):
        """
        Test handling of the `in` comparison
        """
        @jit(forceobj=True)
        def foo(x, y):
            return x in y

        self.assertTrue(foo(1, [0, 1]))
        self.assertTrue(foo(0, [0, 1]))
        self.assertFalse(foo(2, [0, 1]))

        with self.assertRaises(TypeError) as raises:
            foo(None, None)

        self.assertIn("is not iterable", str(raises.exception))

    def test_delitem(self):
        pyfunc = delitem_usecase
        cres = compile_isolated(pyfunc, (), flags=forceobj)
        cfunc = cres.entry_point

        l = [3, 4, 5]
        cfunc(l)
        self.assertPreciseEqual(l, [])
        with self.assertRaises(TypeError):
            cfunc(42)

    def test_starargs_non_tuple(self):
        def consumer(*x):
            return x

        @jit(forceobj=True)
        def foo(x):
            return consumer(*x)

        arg = "ijo"
        got = foo(arg)
        expect = foo.py_func(arg)
        self.assertEqual(got, tuple(arg))
        self.assertEqual(got, expect)


class TestObjectModeInvalidRewrite(TestCase):
    """
    Tests to ensure that rewrite passes didn't affect objmode lowering.
    """

    def _ensure_objmode(self, disp):
        self.assertTrue(disp.signatures)
        self.assertFalse(disp.nopython_signatures)
        return disp

    def test_static_raise_in_objmode_fallback(self):
        """
        Test code based on user submitted issue at
        https://github.com/numba/numba/issues/2159
        """
        def test0(n):
            return n

        def test1(n):
            if n == 0:
                # static raise will fail in objmode if the IR is modified by
                # rewrite pass
                raise ValueError()
            return test0(n)  # trigger objmode fallback

        compiled = jit(test1)
        self.assertEqual(test1(10), compiled(10))
        self._ensure_objmode(compiled)

    def test_static_setitem_in_objmode_fallback(self):
        """
        Test code based on user submitted issue at
        https://github.com/numba/numba/issues/2169
        """

        def test0(n):
            return n

        def test(a1, a2):
            a1 = np.asarray(a1)
            # static setitem here will fail in objmode if the IR is modified by
            # rewrite pass
            a2[0] = 1
            return test0(a1.sum() + a2.sum())   # trigger objmode fallback

        compiled = jit(test)
        args = np.array([3]), np.array([4])
        self.assertEqual(test(*args), compiled(*args))
        self._ensure_objmode(compiled)

    def test_dynamic_func_objmode(self):
        """
        Test issue https://github.com/numba/numba/issues/3355
        """
        func_text = "def func():\n"
        func_text += "    np.array([1,2,3])\n"
        loc_vars = {}
        custom_globals = {'np': np}
        exec(func_text, custom_globals, loc_vars)
        func = loc_vars['func']
        jitted = jit(forceobj=True)(func)
        jitted()


if __name__ == '__main__':
    unittest.main()
