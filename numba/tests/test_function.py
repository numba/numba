from numba import unittest_support as unittest
from numba import njit, function, cfunc
from .support import TestCase

def dump(foo):
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

        def a(i: int) -> int:
            return i + 1

        def b(i: int) -> int:
            return i + 2

        orig_a = a
        a_type = function.fromobject(a)
        a_sig = a_type.signature()
        a = cfunc(a_sig)(a)

        orig_b = b
        b_type = function.fromobject(b)
        b_sig = b_type.signature()
        b = cfunc(b_sig)(b)
        
        @njit
        def foo(f: a_sig):
            r = f(123)
            return f

        @njit
        def bar():
            a(321)
            return a

        #print(bar())
        r = bar()
        dump(bar)
        print('bar RETURNS', r, hex(r))
        print('CORRECT wrapper address={} {}'.format(a._wrapper_address, hex(a._wrapper_address)))
        return
        self.assertEqual(foo(a), bar())
        self.assertEqual(foo(a), foo(a))
        self.assertNotEqual(foo(a), foo(b))
        self.assertEqual(foo(b), foo(b))
        return
        print(a._wrapper_address)
        
        print(foo.inspect_llvm((a_type,)))
        #print(bar.inspect_llvm(()))
        print(function.addr_cfunc_cache.keys())
        print(id(a), id(orig_a))

    def _test_pyfunc_in_out(self):

        def a(i: int) -> int:
            return i + 1

        orig_a = a
        a_type = function.fromobject(a)
        a_sig = a_type.signature()

        @njit
        def foo(f: a_sig):
            return f

        @njit
        def bar():
            return a

        foo.compile((a_type,))
        print(foo.inspect_llvm((a_type,)))
        
        #print(foo(a))
        
    def test_cfunc_in_call(self):

        def a(i: int) -> int:
            return i + 123456

        orig_a = a
        a_type = function.fromobject(a)
        a_sig = a_type.signature()
        a = cfunc(a_sig)(a)

        # make sure that `a` is can be called via its address
        a_addr = a._wrapper_address
        from ctypes import CFUNCTYPE, c_long
        afunc = CFUNCTYPE(c_long)(a_addr)
        self.assertEqual(afunc(c_long(123)), 123456 + 123)

        @njit
        def foo(f: 'int(int)') -> int:
            return f(123)

        self.assertEqual(foo(a), 123456 + 123)

        @njit
        def bar() -> int:
            return a(321)

        #bar_type = function.fromobject(bar)
        #bar_sig = bar_type.signature()
        #bar.compile(bar_sig)
        #print(bar.inspect_llvm(bar_sig.args))
    
        self.assertEqual(bar(), 123456 + 321)
        
        #foo_type = function.fromobject(foo)
        #foo_sig = foo_type.signature()
        #foo.compile(foo_sig)
        #print(foo.inspect_llvm(foo_sig.args))

    
    def test_cfunc_seq(self):

        def a(i: int) -> int:
            return i + 123

        def b(i: int) -> int:
            return i + 456

        orig_a = a
        a_type = function.fromobject(a)
        a_sig = a_type.signature()
        a = cfunc(a_sig)(a)
        b = cfunc(a_sig)(b)

        @njit
        def foo(f: 'int(int)', g: 'int(int)', i: int) -> int:
            s = 0
            seq = (f, g)
            for f_ in seq:
                s += f_(i)
            return s

        self.assertEqual(foo(a, b, 78), 78 + 123 + 78 + 456)

    def test_cfunc_choose(self):

        def a(i: int) -> int:
            return i + 123

        def b(i: int) -> int:
            return i + 456

        orig_a = a
        a_type = function.fromobject(a)
        a_sig = a_type.signature()
        a = cfunc(a_sig)(a)
        b = cfunc(a_sig)(b)

        @njit
        def foo(choose_a: bool) -> int:
            if choose_a:
                f = a
            else:
                f = b
            return f(123)

        self.assertEqual(foo(True), 123 + 123)
        self.assertEqual(foo(False), 123 + 456)
