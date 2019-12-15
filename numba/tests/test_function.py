from numba import njit, function, cfunc
from .support import TestCase


def dump(foo):  # FOR DEBUGGING, TO BE REMOVED
    foo_type = function.fromobject(foo)
    foo_sig = foo_type.signature()
    foo.compile(foo_sig)
    print('{" LLVM IR OF "+foo.__name__+" ":*^70}')
    print(foo.inspect_llvm(foo_sig.args))
    print('{"":*^70}')


class TestFuncionType(TestCase):

    def _test_issue_3405(self):

        @njit
        def a():
            return 2

        @njit
        def b():
            return 3

        def g(arg):
            if arg:
                f = a
            else:
                f = b
            out = f()
            return out

        self.assertEqual(g(True), 2)
        self.assertEqual(g(False), 3)

        print(njit(g).inspect_types())

        self.assertEqual(njit(g)(True), g(True))
        self.assertEqual(njit(g)(False), g(False))

    def test_cfunc_in_out(self):
        """njitted function returns Python functions
        """

        @cfunc('int64(int64)')
        def a(i):
            return i + 1

        @cfunc('int64(int64)')
        def b(i):
            return i + 2

        @njit
        def foo(f):
            f(123)
            return f

        @njit
        def bar():
            a(321)
            return a

        self.assertEqual(bar(), a)
        self.assertEqual(foo(a), a)
        self.assertEqual(foo(b), b)

    def _test_pyfunc_in_out(self):

        @cfunc('int64(int64)')
        def a(i):
            return i + 1

        @njit
        def foo(f):
            return f

        @njit
        def bar():
            return a

    def test_cfunc_in_call(self):

        @cfunc('int64(int64)')
        def a(i):
            return i + 123456

        # make sure that `a` is can be called via its address
        a_addr = a._wrapper_address
        from ctypes import CFUNCTYPE, c_long
        afunc = CFUNCTYPE(c_long)(a_addr)
        self.assertEqual(afunc(c_long(123)), 123456 + 123)

        @njit
        def foo(f):
            return f(123)

        self.assertEqual(foo(a), 123456 + 123)

        @njit
        def bar():
            return a(321)

        self.assertEqual(bar(), 123456 + 321)

    def test_cfunc_seq(self):

        @cfunc('int64(int64)')
        def a(i):
            return i + 123

        @cfunc('int64(int64)')
        def b(i):
            return i + 456

        @njit
        def foo(f, g, i):
            s = 0
            seq = (f, g)
            for f_ in seq:
                s += f_(i)
            return s

        self.assertEqual(foo(a, b, 78), 78 + 123 + 78 + 456)

    def test_cfunc_choose(self):

        @cfunc('int64(int64)')
        def a(i):
            return i + 123

        @cfunc('int64(int64)')
        def b(i):
            return i + 456

        @njit
        def foo(choose_a):
            if choose_a:
                f = a
            else:
                f = b
            return f(123)

        self.assertEqual(foo(True), 123 + 123)
        self.assertEqual(foo(False), 123 + 456)
