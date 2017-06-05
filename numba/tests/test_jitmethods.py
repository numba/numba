from __future__ import absolute_import, print_function, division

import unittest

from numba import jitmethod, jitclass, njit
from .support import TestCase, MemoryLeakMixin, tag


class TestJitMethod(TestCase, MemoryLeakMixin):

    @tag('important')
    def test_signature(self):
        from numba import int32, optional

        spec = [
            ("value", int32),
        ]

        @jitclass(spec)
        class Apple(object):
            def __init__(self, value):
                self.value = value

            @jitmethod((optional(int32),))
            def foo(self, val):
                if val is None:
                    return False

                else:
                    self.value += val
                    return True

        a = Apple(123)
        self.assertEqual(a.value, 123)
        self.assertTrue(a.foo(321))
        self.assertEqual(a.value, 123 + 321)
        self.assertFalse(a.foo(None))
        self.assertEqual(a.value, 123 + 321)

    def test_init(self):
        from numba import int32

        spec = [
            ("value", int32),
        ]

        @jitclass(spec)
        class Apple(object):
            @jitmethod((int32, int32))
            def __init__(self, value, div):
                self.value = value * div

        a = Apple(123, 0.5)  # 0.5 will be truncated to 0
        self.assertEqual(a.value, 0)

    def test_nogil(self):
        from numba import float64

        spec = [
            ("value", float64),
        ]

        @jitclass(spec)
        class Apple(object):
            def __init__(self, value):
                self.value = value

            @jitmethod(nogil=True)
            def method(self):
                return self.value

        @njit(nogil=True)
        def func(self):
            return self.value

        a = Apple(3.14)

        # compile the jitmethod
        a.method()
        # get the dispatcher object of the jitmethod from the class object
        method = Apple.method.dispatcher
        self.assertIs(a.method.dispatcher, method)
        # load the llvm ir
        method_llvm_list = list(method.inspect_llvm().values())
        # ensure it references PyEval_SaveThread to release the GIL
        self.assertIn("PyEval_SaveThread", method_llvm_list[0])
        self.assertEqual(len(method_llvm_list), 1)

        # check the non jitmethod
        func(a)
        func_llvm_list = list(func.inspect_llvm().values())
        # ensure it references PyEval_SaveThread to release the GIL
        self.assertIn("PyEval_SaveThread", func_llvm_list[0])
        self.assertEqual(len(func_llvm_list), 1)


if __name__ == '__main__':
    unittest.main()
