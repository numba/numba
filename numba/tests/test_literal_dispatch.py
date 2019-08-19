from __future__ import print_function


import numba.unittest_support as unittest

from numba.tests.support import TestCase
from numba import njit, types
from numba.special import literally


class TestLiteralDispatcher(TestCase):
    def test_literal_basic(self):
        @njit
        def foo(x):
            return literally(x)

        self.assertEqual(foo(123), 123)
        self.assertEqual(foo(321), 321)

        sig1, sig2 = foo.signatures
        self.assertEqual(sig1[0].literal_value, 123)
        self.assertEqual(sig2[0].literal_value, 321)

    def test_literal_nested(self):
        @njit
        def foo(x):
            return literally(x) * 2

        @njit
        def bar(y, x):
            return foo(y) + x

        y, x = 3, 7
        self.assertEqual(bar(y, x), y * 2 + x)
        [foo_sig] = foo.signatures
        self.assertEqual(foo_sig[0], types.literal(y))
        [bar_sig] = bar.signatures
        self.assertEqual(bar_sig[0], types.literal(y))
        self.assertNotIsInstance(bar_sig[1], types.Literal)


if __name__ == '__main__':
    unittest.main()
