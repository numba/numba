from __future__ import absolute_import, print_function
from numba import unittest_support as unittest
from numba import extensible
from numba import int32, float64
from numba import njit


class TestPlainOldData(unittest.TestCase):
    def test_simple_usecase_from_python(self):
        class Pair(extensible.PlainOldData):
            __fields__ = [
                ('first', int32),
                ('second', float64),
            ]

            def add_both_by(self, val):
                self.first += val
                self.second += val

        pair = Pair()

        self.assertEqual(pair.first, 0)
        self.assertEqual(pair.second, 0)

        pair.first = 1
        pair.second = 2.2

        self.assertEqual(pair.first, 1)
        self.assertAlmostEqual(pair.second, 2.2)

        pair.add_both_by(123)

        self.assertEqual(pair.first, 123 + 1)
        self.assertAlmostEqual(pair.second, 123 + 2.2)

    def test_usecase_in_njit(self):
        class Pair(extensible.PlainOldData):
            __fields__ = [
                ('first', int32),
                ('second', float64),
            ]

            def add_both_by(self, val):
                self.first += val
                self.second += val

        pair = Pair()

        @njit
        def foo(pair):
            a = pair.first + pair.second
            pair.add_both_by(a)

        pair.first = 1
        pair.second = 2

        self.assertEqual(pair.first, 1)
        self.assertEqual(pair.second, 2)

        foo(pair)

        self.assertEqual(pair.first, 1 + 1 + 2)
        self.assertEqual(pair.second, 2 + 1 + 2)


if __name__ == '__main__':
    unittest.main()
