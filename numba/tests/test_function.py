import types as pytypes
from numba import njit, function, cfunc, types, int64, int32
import ctypes

from .support import TestCase


def dump(foo):  # FOR DEBUGGING, TO BE REMOVED
    foo_type = function.fromobject(foo)
    foo_sig = foo_type.signature()
    foo.compile(foo_sig)
    print('{" LLVM IR OF "+foo.__name__+" ":*^70}')
    print(foo.inspect_llvm(foo_sig.args))
    print('{"":*^70}')


# Decorators for tranforming a Python function to different kinds of functions:
def pure_func(func, sig=int64(int64)):
    assert isinstance(func, pytypes.FunctionType), repr(func)
    func.pyfunc = func
    return func


def cfunc_func(func, sig=int64(int64)):
    assert isinstance(func, pytypes.FunctionType), repr(func)
    f = cfunc(sig)(func)
    f.pyfunc = func
    return f


def njit_func(func, sig=int64(int64)):
    assert isinstance(func, pytypes.FunctionType), repr(func)
    f = njit(func)
    f.pyfunc = func
    return f


def njit2_func(func, sig=int64(int64)):
    assert isinstance(func, pytypes.FunctionType), repr(func)
    f = njit(sig)(func)
    f.pyfunc = func
    return f


def ctypes_func(func, sig=int64(int64)):
    assert isinstance(func, pytypes.FunctionType), repr(func)
    cfunc = cfunc_func(func, sig)
    addr = cfunc._wrapper_address
    if sig == int64(int64):
        f = ctypes.CFUNCTYPE(ctypes.c_int64)(addr)
        f.pyfunc = func
        return f
    raise NotImplementedError(
        f'ctypes decorator for {func} with signature {sig}')


class WAP(types.WrapperAddressProtocol):

    def __init__(self, func, sig):
        self.pyfunc = func
        self.cfunc = cfunc_func(func, sig)
        self.sig = sig

    def __wrapper_address__(self, sig):
        assert self.sig == sig, (self.sig, sig)
        return self.cfunc._wrapper_address

    def signature(self):
        return self.sig

    def __call__(self, *args, **kwargs):
        return self.pyfunc(*args, **kwargs)


def wap_func(func, sig=int64(int64)):
    return WAP(func, sig)


all_func_kinds = [pure_func, cfunc_func, njit_func,
                  njit2_func, ctypes_func, wap_func]


class TestFuncionType(TestCase):

    def _test_issue_3405_original(self):

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

        self.assertEqual(njit(g)(True), g(True))
        self.assertEqual(njit(g)(False), g(False))

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

    def test_in(self):

        def a(i):
            return i + 1

        def foo(f):
            return 0

        for decor in all_func_kinds:
            if decor in [pure_func]:
                # Skip not-yet-supported functions
                continue
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                self.assertEqual(njit(foo)(a_), foo(a))

    def test_in_call(self):

        def a(i):
            return i + 1

        def foo(f):
            r = f(123)
            return r

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func]:
                # Skip not-yet-supported functions
                continue
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                self.assertEqual(njit(foo)(a_), foo(a))

    def test_in_call_out(self):

        def a(i):
            return i + 1

        def foo(f):
            f(123)
            return f

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func, njit_func, njit2_func,
                         wap_func]:
                # Skip not-yet-supported functions
                continue
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                if decor is cfunc_func:
                    self.assertEqual(njit(foo)(a_).pyfunc, foo(a))
                else:
                    self.assertEqual(njit(foo)(a_), foo(a))

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

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func, njit_func, njit2_func]:
                # Skip not-yet-supported functions
                continue
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

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func, njit_func, njit2_func,
                         wap_func]:
                # Skip not-yet-supported functions
                continue
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                b_ = decor(b)
                self.assertEqual(njit(mkfoo(b_))(a_), mkfoo(b)(a))

    def test_ns_call(self):

        def a(i):
            return i + 1

        def mkfoo(a_):
            def foo():
                r = a_(123)
                return r
            return foo

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func, wap_func]:
                # Skip not-yet-supported functions
                continue
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

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func, njit_func, njit2_func,
                         wap_func]:
                # Skip not-yet-supported functions
                continue
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

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func, njit_func, njit2_func,
                         wap_func]:
                # Skip not-yet-supported functions
                continue
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

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func, njit2_func, wap_func,
                         cfunc_func]:
                # Skip not-yet-supported functions
                continue
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

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func, njit2_func, wap_func,
                         cfunc_func]:
                # Skip not-yet-supported functions
                continue
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

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func]:
                # Skip not-yet-supported functions
                continue
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

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func, wap_func]:
                # Skip not-yet-supported functions
                continue
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

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func, wap_func, njit_func,
                         njit2_func]:
                # Skip not-yet-supported functions
                continue
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

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func, njit_func, njit2_func]:
                # Skip not-yet-supported functions
                continue
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

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func, njit_func, njit2_func]:
                # Skip not-yet-supported functions
                continue
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

        for decor in all_func_kinds:
            if decor in [pure_func, ctypes_func, njit_func, njit2_func]:
                # Skip not-yet-supported functions
                continue
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                b_ = decor(b)
                self.assertEqual(njit(foo)((a_, b_), 2), foo((a, b), 2))


class TestFuncionTypeExtensions(TestCase):

    def _test_wrapper_address_protocol(self):
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
                    print(f'libpath={libpath}')
                    if 1 or sys.platform == 'darwin':
                        libc = ctypes.cdll.LoadLibrary(libpath)
                    else:
                        # TODO: find libc.so in a more portable way
                        libc = ctypes.CDLL(libpath)
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
