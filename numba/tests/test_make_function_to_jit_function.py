from __future__ import print_function, division, absolute_import

from numba import njit
from numba.extending import overload
import numpy as np

from numba import unittest_support as unittest


@njit
def consumer(func, *args):
    return func(*args)


@njit
def consumer2arg(func1, func2):
    return func2(func1)


_global = 123


class TestMakeFunctionToJITFunction(unittest.TestCase):
    """
    This tests the pass that converts ir.Expr.op == make_function (i.e. closure)
    into a JIT function.
    """
    # NOTE: testing this is a bit tricky. The function receiving a JIT'd closure
    # must also be under JIT control so as to handle the JIT'd closure
    # correctly, however, in the case of running the test implementations in the
    # interpreter, the receiving function cannot be JIT'd else it will receive
    # the Python closure and then complain about pyobjects as arguments.
    # The way around this is to use a factory function to close over either the
    # jitted or standard python function as the consumer depending on context.

    def test_escape(self):

        def impl_factory(consumer_func):
            def impl():
                def inner():
                    return 10
                return consumer_func(inner)
            return impl

        cfunc = njit(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)

        self.assertEqual(impl(), cfunc())

    def test_nested_escape(self):

        def impl_factory(consumer_func):
            def impl():
                def inner():
                    return 10

                def innerinner(x):
                    return x()
                return consumer_func(inner, innerinner)
            return impl

        cfunc = njit(impl_factory(consumer2arg))
        impl = impl_factory(consumer2arg.py_func)

        self.assertEqual(impl(), cfunc())

    def test_closure_in_escaper(self):

        def impl_factory(consumer_func):
            def impl():
                def callinner():
                    def inner():
                        return 10
                    return inner()
                return consumer_func(callinner)
            return impl

        cfunc = njit(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)

        self.assertEqual(impl(), cfunc())

    def test_close_over_consts(self):

        def impl_factory(consumer_func):
            def impl():
                y = 10

                def callinner(z):
                    return y + z + _global
                return consumer_func(callinner, 6)
            return impl

        cfunc = njit(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)

        self.assertEqual(impl(), cfunc())

    def test_close_over_consts_w_args(self):

        def impl_factory(consumer_func):
            def impl(x):
                y = 10

                def callinner(z):
                    return y + z + _global
                return consumer_func(callinner, x)
            return impl

        cfunc = njit(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)

        a = 5
        self.assertEqual(impl(a), cfunc(a))

    def test_with_overload(self):

        def foo(func, *args):
            nargs = len(args)
            if nargs == 1:
                return func(*args)
            elif nargs == 2:
                return func(func(*args))

        @overload(foo)
        def foo_ol(func, *args):
            # specialise on the number of args, as per `foo`
            nargs = len(args)
            if nargs == 1:
                def impl(func, *args):
                    return func(*args)
                return impl
            elif nargs == 2:
                def impl(func, *args):
                    return func(func(*args))
                return impl

        def impl_factory(consumer_func):
            def impl(x):
                y = 10

                def callinner(*z):
                    return y + np.sum(np.asarray(z)) + _global
                # run both specialisations, 1 arg, and 2 arg.
                return foo(callinner, x), foo(callinner, x, x)
            return impl

        cfunc = njit(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)

        a = 5
        self.assertEqual(impl(a), cfunc(a))

    def test_basic_apply_like_case(self):
        def apply(array, func):
            return func(array)

        @overload(apply)
        def ov_apply(array, func):
            return lambda array, func: func(array)

        def impl(array):
            def mul10(x):
                return x * 10
            return apply(array, mul10)

        cfunc = njit(impl)

        a = np.arange(10)
        np.testing.assert_allclose(impl(a), cfunc(a))

    @unittest.skip("Needs option/flag inheritance to work")
    def test_jit_option_inheritance(self):

        def impl_factory(consumer_func):
            def impl(x):
                def inner(val):
                    return 1 / val
                return consumer_func(inner, x)
            return impl

        cfunc = njit(error_model='numpy')(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)

        a = 0
        self.assertEqual(impl(a), cfunc(a))


if __name__ == '__main__':
    unittest.main()
