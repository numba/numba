from __future__ import absolute_import, print_function, division

import unittest

from numba import jitmethod, jitclass
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

    @tag('important')
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


if __name__ == '__main__':
    unittest.main()
