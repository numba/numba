import types as pytypes
from numba import njit, function, cfunc, types
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
def pure_func(func, sig='int64(int64)'):
    assert isinstance(func, pytypes.FunctionType), repr(func)
    func.pyfunc = func
    return func


def cfunc_func(func, sig='int64(int64)'):
    assert isinstance(func, pytypes.FunctionType), repr(func)
    f = cfunc(sig)(func)
    f.pyfunc = func
    return f


def njit_func(func, sig='int64(int64)'):
    assert isinstance(func, pytypes.FunctionType), repr(func)
    f = njit(func)
    f.pyfunc = func
    return f


def njit2_func(func, sig='int64(int64)'):
    assert isinstance(func, pytypes.FunctionType), repr(func)
    f = njit(sig)(func)
    f.pyfunc = func
    return f


def ctypes_func(func, sig='int64(int64)'):
    assert isinstance(func, pytypes.FunctionType), repr(func)
    cfunc = cfunc_func(func, sig)
    addr = cfunc._wrapper_address
    if sig == 'int64(int64)':
        f = ctypes.CFUNCTYPE(ctypes.c_int64)(addr)
        f.pyfunc = func
        return f
    # TODO: numbatype(sig) to ctypes converter, see RBC for an example
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


def wap_func(func, sig='int64(int64)'):
    return WAP(func, sig)


all_func_kinds = [pure_func, cfunc_func, njit_func,
                  njit2_func, ctypes_func, wap_func]
supported_func_kinds = [cfunc_func]


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

    def _test_in(self, decor):

        @decor
        def a(i):
            return i + 1

        def foo(f):
            return 0

        self.assertEqual(njit(foo)(a), foo(a.pyfunc))

    def _test_in_call(self, decor):

        @decor
        def a(i):
            return i + 1

        def foo(f):
            r = f(123)
            return r

        self.assertEqual(njit(foo)(a), foo(a.pyfunc))

    def _test_in_call_out(self, decor):

        @decor
        def a(i):
            return i + 1

        def foo(f):
            f(123)
            return f

        self.assertEqual(njit(foo)(a), foo(a.pyfunc))

    def _test_in_seq_call(self, decor):

        @decor
        def a(i):
            return i + 1

        @decor
        def b(i):
            return i + 2

        def foo(f, g):
            r = 0
            for f_ in (f, g):
                r = r + f_(r)
            return r

        self.assertEqual(njit(foo)(a, b), foo(a.pyfunc, b.pyfunc))

    def _test_in_ns_seq_call(self, decor):

        @decor
        def a(i):
            return i + 1

        @decor
        def b(i):
            return i + 2

        def w(op, a_, b_):
            def foo(f):
                r = 0
                for f_ in (f, b_):
                    r = r + f_(r)
                return r
            return op(foo)(a_)

        self.assertEqual(w(njit, a, b), w(lambda f:f, a.pyfunc, b.pyfunc))

    def _test_ns_call(self, decor):

        @decor
        def a(i):
            return i + 1

        def foo():
            r = a(123)
            return r

        self.assertEqual(njit(foo)(), a.pyfunc(123))

    def _test_ns_out(self, decor):

        @decor
        def a(i):
            return i + 1

        def foo():
            return a

        self.assertEqual(njit(foo)(), a)

    def _test_ns_call_out(self, decor):

        @decor
        def a(i):
            return i + 1

        def foo():
            a(123)
            return a

        self.assertEqual(njit(foo)(), a)

    def _test_in_overload(self, decor):

        @decor
        def a(i):
            return i + 1

        def foo(f):
            r1 = f(123)
            r2 = f(123.45)
            return (r1, r2)

        self.assertEqual(njit(foo)(a), foo(a.pyfunc))

    def _test_ns_overload(self, decor):

        @decor
        def a(i):
            return i + 1

        def w(op, a):
            def foo():
                r1 = a(123)
                r2 = a(123.45)
                return (r1, r2)
            return op(foo)()

        self.assertEqual(w(njit, a), w(lambda f:f, a.pyfunc))

    def _test_in_choose(self, decor):

        @decor
        def a(i):
            return i + 1

        @decor
        def b(i):
            return i + 2

        def w(op, a, b, choose_left):
            def foo(a, b, choose_left):
                if choose_left:
                    r = a(1)
                else:
                    r = b(2)
                return r
            return op(foo)(a, b, choose_left)

        self.assertEqual(w(njit, a, b, True),
                         w(lambda f:f, a.pyfunc, b.pyfunc, True))
        self.assertEqual(w(njit, a, b, False),
                         w(lambda f:f, a.pyfunc, b.pyfunc, False))
        self.assertNotEqual(w(njit, a, b, True),
                            w(lambda f:f, a.pyfunc, b.pyfunc, False))

    def _test_ns_choose(self, decor):

        @decor
        def a(i):
            return i + 1

        @decor
        def b(i):
            return i + 2

        def w(op, a, b, choose_left):
            def foo(choose_left):
                if choose_left:
                    r = a(1)
                else:
                    r = b(2)
                return r
            return op(foo)(choose_left)

        self.assertEqual(w(njit, a, b, True),
                         w(lambda f:f, a.pyfunc, b.pyfunc, True))
        self.assertEqual(w(njit, a, b, False),
                         w(lambda f:f, a.pyfunc, b.pyfunc, False))
        self.assertNotEqual(w(njit, a, b, True),
                            w(lambda f:f, a.pyfunc, b.pyfunc, False))

    def _test_in_choose_out(self, decor):

        @decor
        def a(i):
            return i + 1

        @decor
        def b(i):
            return i + 2

        def w(op, a, b, choose_left):
            def foo(a, b, choose_left):
                if choose_left:
                    return a
                else:
                    return b
            return op(foo)(a, b, choose_left)

        self.assertEqual(w(njit, a, b, True), w(lambda f:f, a, b, True))
        self.assertEqual(w(njit, a, b, False), w(lambda f:f, a, b, False))
        self.assertNotEqual(w(njit, a, b, True), w(lambda f:f, a, b, False))

    def _test_in_choose_func(self, decor):

        @decor
        def a(i):
            return i + 1

        @decor
        def b(i):
            return i + 2

        def w(op, a, b, choose_left):
            def foo(a, b, choose_left):
                if choose_left:
                    f = a
                else:
                    f = b
                return f(1)
            return op(foo)(a, b, choose_left)

        self.assertEqual(w(njit, a, b, True),
                         w(lambda f:f, a.pyfunc, b.pyfunc, True))
        self.assertEqual(w(njit, a, b, False),
                         w(lambda f:f, a.pyfunc, b.pyfunc, False))
        self.assertNotEqual(w(njit, a, b, True),
                            w(lambda f:f, a.pyfunc, b.pyfunc, False))

    def _test_in_pick_func_call(self, decor):

        @decor
        def a(i):
            return i + 1

        @decor
        def b(i):
            return i + 2

        def w(op, a, b, index):
            def foo(funcs, i):
                r = funcs[i](123)
                return r
            return op(foo)((a, b), index)

        self.assertEqual(w(njit, a, b, 0), w(lambda f:f, a.pyfunc, b.pyfunc, 0))
        self.assertEqual(w(njit, a, b, 1), w(lambda f:f, a.pyfunc, b.pyfunc, 1))
        self.assertNotEqual(w(njit, a, b, 0),
                            w(lambda f:f, a.pyfunc, b.pyfunc, 1))

    def _test_in_iter_func_call(self, decor):

        @decor
        def a(i):
            return i + 1

        @decor
        def b(i):
            return i + 2

        def w(op, a, b):
            def foo(funcs, n):
                r = 0
                for i in range(n):
                    f = funcs[i]
                    r = r + f(r)
                return r
            return op(foo)((a, b), 2)

        self.assertEqual(w(njit, a, b), w(lambda f:f, a.pyfunc, b.pyfunc))

    def test_all(self):
        test_methods = [
            self._test_in, self._test_in_call, self._test_in_call_out,
            self._test_ns_call, self._test_ns_out, self._test_ns_call_out,
            self._test_in_seq_call, self._test_in_ns_seq_call,
            self._test_in_overload, self._test_ns_overload,
            self._test_in_choose, self._test_ns_choose,
            self._test_in_choose_out, self._test_in_choose_func,
            self._test_in_pick_func_call, self._test_in_iter_func_call
        ]
        count = 0
        success = 0
        for mth in test_methods:
            for decor in all_func_kinds:
                count += 1
                try:
                    mth(decor)
                except Exception as msg:
                    msgline = str(msg).splitlines(1)[0].strip()
                    print(f'{mth.__name__}[{decor.__name__}] support failed:'
                          f' {msgline}')
                else:
                    success += 1
                    print(f'{mth.__name__}[{decor.__name__}] support works OK')
        print(f'test_all success rate: {success}/{count}')


class TestFuncionTypeSupport(TestCase):

    def test_numbatype(self):
        worker = types.function.numbatype
        cptr = types.CPointer

        def foo(i: int) -> int:
            pass

        for target_type, type_sources in [
                # primitive types
                (types.boolean, ['bool', 'boolean', bool, types.boolean, 'b1']),
                (types.none, ['void', types.none]),
                (types.int8, ['int8', types.int8, 'i1']),
                (types.int16, ['int16', types.int16, 'i2']),
                (types.int32, ['int32', types.int32, 'i4']),
                (types.int64, ['int64', int, types.int64, 'i8']),
                (types.uint8, ['uint8', types.uint8, 'u1', 'byte']),
                (types.uint16, ['uint16', types.uint16, 'u2']),
                (types.uint32, ['uint32', types.uint32, 'u4']),
                (types.uint64, ['uint64', types.uint64, 'u8']),
                (types.float32, ['float32', 'float', types.float32, 'f4']),
                (types.float64,
                 ['float64', 'double', float, types.float64, 'f8']),
                (types.complex64,
                 ['complex64', 'complex', types.complex64, 'c8']),
                (types.complex128,
                 ['complex128', complex, types.complex128, 'c16']),
                (types.unicode_type,
                 ['str', str, types.unicode_type, 'unicode', 'string']),
                (types.none, ['void', 'none', types.void, types.none]),
                (types.voidptr, [types.voidptr, 'void*']),
                (types.pyobject, [types.pyobject]),
                (types.pyfunc_type, [types.pyfunc_type]),
                (types.slice2_type, [types.slice2_type]),
                (types.slice3_type, [types.slice3_type]),
                (types.code_type, [types.code_type]),
                (types.undefined, [types.undefined]),
                (types.Any, [types.Any]),
                (types.range_iter32_type, [types.range_iter32_type]),
                (types.range_iter64_type, [types.range_iter64_type]),
                (types.unsigned_range_iter64_type,
                 [types.unsigned_range_iter64_type]),
                (types.range_state32_type, [types.range_state32_type]),
                (types.range_state64_type, [types.range_state64_type]),
                (types.ellipsis, [types.ellipsis]),
                # composite types
                (cptr(types.int64), ['int64*', 'i8 *']),
                (types.FunctionType((types.int64, (types.int64,))),
                 ['int64(int64)', 'i8(i8)', 'int(i8)',
                  types.int64(types.int64), foo]),
        ]:
            for source_type in type_sources:
                self.assertEqual(
                    target_type, worker(source_type),
                    msg=(f'expected {target_type} ({type(target_type)})'
                         f' from {source_type} ({type(source_type)})'))


class TestFuncionTypeExtensions(TestCase):

    def test_wrapper_address_protocol(self):
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
                if (self.fname, sig) == ('time', 'int32()'):
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
                    return 'int32()'
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
