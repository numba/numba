"""
Tests for numba.types.
"""

from __future__ import print_function, absolute_import

from numba.utils import IS_PY3
from numba import types
from numba import unittest_support as unittest


class TestTypeNames(unittest.TestCase):

    def test_equality(self):
        self.assertEqual(types.int32, types.int32)
        self.assertEqual(types.uint32, types.uint32)
        self.assertEqual(types.complex64, types.complex64)
        self.assertEqual(types.float32, types.float32)
        # Different signedness
        self.assertNotEqual(types.int32, types.uint32)
        # Different width
        self.assertNotEqual(types.int64, types.int32)
        self.assertNotEqual(types.float64, types.float32)
        self.assertNotEqual(types.complex64, types.complex128)
        # Different domain
        self.assertNotEqual(types.int64, types.float64)
        self.assertNotEqual(types.uint64, types.float64)
        self.assertNotEqual(types.complex64, types.float64)

    def test_ordering(self):
        def check_order(values):
            for i in range(len(values)):
                self.assertLessEqual(values[i], values[i])
                self.assertGreaterEqual(values[i], values[i])
                self.assertFalse(values[i] < values[i])
                self.assertFalse(values[i] > values[i])
                for j in range(i):
                    self.assertLess(values[j], values[i])
                    self.assertLessEqual(values[j], values[i])
                    self.assertGreater(values[i], values[j])
                    self.assertGreaterEqual(values[i], values[j])
                    self.assertFalse(values[i] < values[j])
                    self.assertFalse(values[i] <= values[j])
                    self.assertFalse(values[j] > values[i])
                    self.assertFalse(values[j] >= values[i])

        check_order([types.int8, types.int16, types.int32, types.int64])
        check_order([types.uint8, types.uint16, types.uint32, types.uint64])
        check_order([types.float32, types.float64])
        check_order([types.complex64, types.complex128])

        if IS_PY3:
            with self.assertRaises(TypeError):
                types.int8 <= types.uint32
            with self.assertRaises(TypeError):
                types.int8 <= types.float32
            with self.assertRaises(TypeError):
                types.float64 <= types.complex128


if __name__ == '__main__':
    unittest.main()
