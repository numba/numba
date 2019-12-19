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

    def test_njit_in_call(self):

        @njit
        def a(i):
            return i + 123456

        @njit
        def foo(f):
            return f(123)

        self.assertEqual(foo(a), 123456 + 123)

    def _test_pyfunc_in_call(self):
        # disabled as the pure python function support infers badly
        # with other numba tests, see numba/function.py.

        def a(i):
            return i + 123456

        @njit
        def foo(f):
            return f(123)

        @njit
        def bar(f):
            return f(123.45)

        self.assertEqual(foo(a), 123456 + 123)
        self.assertEqual(bar(a), 123456 + 123.45)

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
        import sys
        import time
        from numba.types import WrapperAddressProtocol

        class LibC(WrapperAddressProtocol):

            def __init__(self, fname):
                if sys.platform[:5] == 'linux':
                    self.libc = ctypes.CDLL("libc.so.6")
                else:
                    raise NotImplementedError(
                        f'loading libc on platform {sys.platform}')
                self.fname = fname

            def __wrapper_address__(self, sig):
                if (self.fname, sig) == ('time', 'int32()'):
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
