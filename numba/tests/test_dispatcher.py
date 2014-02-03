from __future__ import print_function, division, absolute_import
from numba import unittest_support as unittest
from numba.special import typeof
from numba import types
import numpy


class TestDispatcher(unittest.TestCase):
    def test_typeof(self):
        self.assertEqual(typeof(numpy.int8(1)), types.int8)
        self.assertEqual(typeof(numpy.uint16(1)), types.uint16)
        self.assertEqual(typeof(numpy.float64(1)), types.float64)
        self.assertEqual(typeof(numpy.complex128(1)), types.complex128)


if __name__ == '__main__':
    unittest.main()
