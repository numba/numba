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


class TestImmutablePOD(unittest.TestCase):
    def test_simple_usecase_from_python(self):
        class Pair(extensible.ImmutablePOD):
            __fields__ = [
                ('first', int32),
                ('second', float64),
            ]

            def combine(self, val):
                return self.first + self.second + val

        pair = Pair(first=2, second=3)
        self.assertEqual(pair.first, 2)
        self.assertEqual(pair.second, 3)
        self.assertEqual(pair.combine(123), 123 + 2 + 3)

    def test_nopython_usecase(self):
        class Pair(extensible.ImmutablePOD):
            __fields__ = [
                ('first', int32),
                ('second', float64),
            ]

            def combine(self, val):
                return self.first + self.second + val

        pair = Pair(first=2, second=3)

        @njit
        def foo(pair):
            return pair.combine(123)

        self.assertEqual(foo(pair), 123 + 2 + 3)


if __name__ == '__main__':
    unittest.main()
