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


if __name__ == '__main__':
    unittest.main()
