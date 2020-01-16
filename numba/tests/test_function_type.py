import types as pytypes
from numba import njit, function, cfunc, types, int64, int32, float64, float32
import ctypes

from .support import TestCase


def dump(foo):  # FOR DEBUGGING, TO BE REMOVED
    foo_type = function.fromobject(foo)
    foo_sig = foo_type.signature()
    foo.compile(foo_sig)
    print('{" LLVM IR OF "+foo.__name__+" ":*^70}')
    print(foo.inspect_llvm(foo_sig.args))
    print('{"":*^70}')


# Decorators for transforming a Python function to different kinds of
# functions:

def mk_cfunc_func(sig):
    def cfunc_func(func):
        assert isinstance(func, pytypes.FunctionType), repr(func)
        f = cfunc(sig)(func)
        f.pyfunc = func
        return f
    return cfunc_func


def njit_func(func):
    assert isinstance(func, pytypes.FunctionType), repr(func)
    f = njit(func)
    f.pyfunc = func
    return f


def mk_njit_with_sig_func(sig):
    def njit_with_sig_func(func):
        assert isinstance(func, pytypes.FunctionType), repr(func)
        f = njit(sig)(func)
        f.pyfunc = func
        return f
    return njit_with_sig_func


def mk_ctypes_func(sig):
    def ctypes_func(func, sig=int64(int64)):
        assert isinstance(func, pytypes.FunctionType), repr(func)
        cfunc = mk_cfunc_func(sig)(func)
        addr = cfunc._wrapper_address
        if sig == int64(int64):
            f = ctypes.CFUNCTYPE(ctypes.c_int64)(addr)
            f.pyfunc = func
            return f
        raise NotImplementedError(
            f'ctypes decorator for {func} with signature {sig}')
    return ctypes_func


class WAP(types.WrapperAddressProtocol):

    def __init__(self, func, sig):
        self.pyfunc = func
        self.cfunc = cfunc(sig)(func)
        self.sig = sig

    def __wrapper_address__(self, sig):
        assert self.sig == sig, (self.sig, sig)
        return self.cfunc._wrapper_address

    def signature(self):
        return self.sig

    def __call__(self, *args, **kwargs):
        return self.pyfunc(*args, **kwargs)


def mk_wap_func(sig):
    def wap_func(func):
        return WAP(func, sig)
    return wap_func


class TestFunctionType(TestCase):

    def test_in(self):

        def a(i):
            return i + 1

        def foo(f):
            return 0

        sig = int64(int64)

        for decor in [mk_cfunc_func(sig), njit_func,
                      mk_njit_with_sig_func(sig), mk_ctypes_func(sig),
                      mk_wap_func(sig)]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                self.assertEqual(njit(foo)(a_), foo(a))

    def test_in_call(self):

        def a(i):
            return i + 1

        def foo(f):
            return f(123)

        sig = int64(int64)

        for decor in [mk_cfunc_func(sig), njit_func,
                      mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                self.assertEqual(njit(foo)(a_), foo(a))

    def test_in_call_out(self):

        def a(i):
            return i + 1

        def foo(f):
            f(123)
            return f

        sig = int64(int64)

        for decor in [mk_cfunc_func(sig)]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                self.assertEqual(njit(foo)(a_).pyfunc, foo(a))

    def test_in_seq_call(self):

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def foo(f, g):
            r = 0
            for f_ in (f, g):
                r = r + f_(r)
            return r

        sig = int64(int64)

        for decor in [mk_cfunc_func(sig), mk_wap_func(sig)]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                b_ = decor(b)
                self.assertEqual(njit(foo)(a_, b_), foo(a, b))

    def test_in_ns_seq_call(self):

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def mkfoo(b_):
            def foo(f):
                r = 0
                for f_ in (f, b_):
                    r = r + f_(r)
                return r
            return foo

        sig = int64(int64)

        for decor in [mk_cfunc_func(sig)]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                b_ = decor(b)
                self.assertEqual(njit(mkfoo(b_))(a_), mkfoo(b)(a))

    def test_ns_call(self):

        def a(i):
            return i + 1

        def mkfoo(a_):
            def foo():
                return a_(123)
            return foo

        sig = int64(int64)

        for decor in [mk_cfunc_func(sig), njit_func,
                      mk_njit_with_sig_func(sig)]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                self.assertEqual(njit(mkfoo(a_))(), mkfoo(a)())

    def test_ns_out(self):

        def a(i):
            return i + 1

        def mkfoo(a_):
            def foo():
                return a_
            return foo

        sig = int64(int64)

        for decor in [mk_cfunc_func(sig)]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                self.assertEqual(njit(mkfoo(a_))().pyfunc, mkfoo(a)())

    def test_ns_call_out(self):

        def a(i):
            return i + 1

        def mkfoo(a_):
            def foo():
                a_(123)
                return a_
            return foo

        sig = int64(int64)

        for decor in [mk_cfunc_func(sig)]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                self.assertEqual(njit(mkfoo(a_))().pyfunc, mkfoo(a)())

    def test_in_overload(self):

        def a(i):
            return i + 1

        def foo(f):
            r1 = f(123)
            r2 = f(123.45)
            return (r1, r2)

        for decor in [njit_func]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                self.assertEqual(njit(foo)(a_), foo(a))

    def test_ns_overload(self):

        def a(i):
            return i + 1

        def mkfoo(a_):
            def foo():
                r1 = a_(123)
                r2 = a_(123.45)
                return (r1, r2)
            return foo

        for decor in [njit_func]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                self.assertEqual(njit(mkfoo(a_))(), mkfoo(a)())

    def test_in_choose(self):

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def foo(a, b, choose_left):
            if choose_left:
                r = a(1)
            else:
                r = b(2)
            return r

        sig = int64(int64)

        for decor in [mk_cfunc_func(sig), njit_func,
                      mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                b_ = decor(b)
                self.assertEqual(njit(foo)(a_, b_, True), foo(a, b, True))
                self.assertEqual(njit(foo)(a_, b_, False), foo(a, b, False))
                self.assertNotEqual(njit(foo)(a_, b_, True), foo(a, b, False))

    def test_ns_choose(self):

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def mkfoo(a_, b_):
            def foo(choose_left):
                if choose_left:
                    r = a_(1)
                else:
                    r = b_(2)
                return r
            return foo

        sig = int64(int64)

        for decor in [mk_cfunc_func(sig), njit_func,
                      mk_njit_with_sig_func(sig)]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                b_ = decor(b)
                self.assertEqual(njit(mkfoo(a_, b_))(True),
                                 mkfoo(a, b)(True))
                self.assertEqual(njit(mkfoo(a_, b_))(False),
                                 mkfoo(a, b)(False))
                self.assertNotEqual(njit(mkfoo(a_, b_))(True),
                                    mkfoo(a, b)(False))

    def test_in_choose_out(self):

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def foo(a, b, choose_left):
            if choose_left:
                return a
            else:
                return b

        sig = int64(int64)

        for decor in [mk_cfunc_func(sig)]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                b_ = decor(b)
                self.assertEqual(njit(foo)(a_, b_, True).pyfunc,
                                 foo(a, b, True))
                self.assertEqual(njit(foo)(a_, b_, False).pyfunc,
                                 foo(a, b, False))
                self.assertNotEqual(njit(foo)(a_, b_, True).pyfunc,
                                    foo(a, b, False))

    def test_in_choose_func_value(self):

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def foo(a, b, choose_left):
            if choose_left:
                f = a
            else:
                f = b
            return f(1)

        sig = int64(int64)

        for decor in [mk_cfunc_func(sig), mk_wap_func(sig)]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                b_ = decor(b)
                self.assertEqual(njit(foo)(a_, b_, True), foo(a, b, True))
                self.assertEqual(njit(foo)(a_, b_, False), foo(a, b, False))
                self.assertNotEqual(njit(foo)(a_, b_, True), foo(a, b, False))

    def test_in_pick_func_call(self):

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def foo(funcs, i):
            r = funcs[i](123)
            return r

        sig = int64(int64)

        for decor in [mk_cfunc_func(sig), mk_wap_func(sig)]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                b_ = decor(b)
                self.assertEqual(njit(foo)((a_, b_), 0), foo((a, b), 0))
                self.assertEqual(njit(foo)((a_, b_), 1), foo((a, b), 1))
                self.assertNotEqual(njit(foo)((a_, b_), 0), foo((a, b), 1))

    def test_in_iter_func_call(self):

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def foo(funcs, n):
            r = 0
            for i in range(n):
                f = funcs[i]
                r = r + f(r)
            return r

        sig = int64(int64)

        for decor in [mk_cfunc_func(sig), mk_wap_func(sig)]:
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                b_ = decor(b)
                self.assertEqual(njit(foo)((a_, b_), 2), foo((a, b), 2))


class TestFunctionTypeExtensions(TestCase):

    def test_wrapper_address_protocol_libm(self):
        import os
        import ctypes.util
        from numba.types import WrapperAddressProtocol

        class LibM(WrapperAddressProtocol):

            def __init__(self, fname):
                if os.name == 'nt':
                    lib = ctypes.cdll.msvcrt
                else:
                    libpath = ctypes.util.find_library('m')
                    lib = ctypes.cdll.LoadLibrary(libpath)
                self.lib = lib
                self.fname = fname

            def __wrapper_address__(self, sig):
                if (self.fname, sig) == ('cos', float64(float64)):
                    addr = ctypes.cast(self.lib.cos, ctypes.c_voidp).value
                elif (self.fname, sig) == ('sinf', float32(float32)):
                    addr = ctypes.cast(self.lib.sinf, ctypes.c_voidp).value
                else:
                    raise NotImplementedError(
                        f'wrapper address of `{self.fname}`'
                        f' with signature `{sig}`')
                return addr

            def signature(self):
                if self.fname == 'cos':
                    return float64(float64)
                if self.fname == 'sinf':
                    return float32(float32)
                raise NotImplementedError(f'signature of `{self.fname}`')

        mycos = LibM('cos')
        mysin = LibM('sinf')

        @njit
        def myeval(f, x):
            return f(x)

        self.assertEqual(myeval(mycos, 0.0), 1.0)
        self.assertEqual(myeval(mysin, float32(0.0)), 0.0)

    def _test_wrapper_address_protocol_libc(self):
        import os
        import sys
        import time
        import ctypes.util
        from numba.types import WrapperAddressProtocol

        class LibC(WrapperAddressProtocol):

            def __init__(self, fname):
                libc = None
                if os.name == 'nt':
                    libc = ctypes.cdll.msvcrt
                elif os.name == 'posix':
                    libpath = ctypes.util.find_library('c')
                    libc = ctypes.cdll.LoadLibrary(libpath)
                if libc is None:
                    raise NotImplementedError(
                        f'loading libc on platform {sys.platform}')
                self.libc = libc
                self.fname = fname

            def __wrapper_address__(self, sig):
                if (self.fname, sig) == ('time', int32()):
                    assert self.libc.time.restype == ctypes.c_int, (
                        self.libc.time.restype, ctypes.c_int)
                    assert ctypes.sizeof(ctypes.c_int) == 4, (
                        ctypes.sizeof(ctypes.c_int), 4)
                    addr = ctypes.cast(self.libc.time, ctypes.c_voidp).value
                else:
                    raise NotImplementedError(
                        f'wrapper address of `{self.fname}`'
                        f' with signature `{sig}`')
                return addr

            def signature(self):
                if self.fname == 'time':
                    return int32()
                raise NotImplementedError(f'signature of `{self.fname}`')

        wap = LibC('time')

        @njit
        def get_time(f):
            return f()

        t0 = time.time()
        # libc.time returns int, so make sure t1 will be ahead of t0
        # at least 1 second:
        time.sleep(1.01)
        t1 = get_time(wap)
        t2 = time.time()

        self.assertLess(t0, t1)
        self.assertLess(t1, t2)


class TestMiscIssues(TestCase):

    def test_issue_3405_using_cfunc(self):

        @cfunc('int64()')
        def a():
            return 2

        @cfunc('int64()')
        def b():
            return 3

        def g(arg):
            if arg:
                f = a
            else:
                f = b
            out = f()
            return out

        self.assertEqual(njit(g)(True), 2)
        self.assertEqual(njit(g)(False), 3)

    def test_pr4967_example(self):

        @cfunc('int64(int64)')
        def a(i):
            return i + 1

        @cfunc('int64(int64)')
        def b(i):
            return i + 2

        @njit
        def foo(f, g):
            i = f(2)
            seq = (f, g)
            for fun in seq:
                i += fun(i)
            return i

        a_ = a._pyfunc
        b_ = b._pyfunc
        self.assertEqual(foo(a, b),
                         a_(2) + a_(a_(2)) + b_(a_(2) + a_(a_(2))))
