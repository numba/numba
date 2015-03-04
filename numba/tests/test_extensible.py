from __future__ import absolute_import, print_function
from numba import unittest_support as unittest
from numba import extensible
from numba import int32, float64


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


if __name__ == '__main__':
    unittest.main()
