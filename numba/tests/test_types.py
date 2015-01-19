"""
Tests for numba.types.
"""

from __future__ import print_function, absolute_import

import gc
import weakref

from numba.utils import IS_PY3
from numba import types
from numba import unittest_support as unittest


class Dummy(object):
    pass


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

    def test_weaktype(self):
        d = Dummy()
        e = Dummy()
        a = types.Dispatcher(d)
        b = types.Dispatcher(d)
        c = types.Dispatcher(e)
        self.assertIs(a.overloaded, d)
        self.assertIs(b.overloaded, d)
        self.assertIs(c.overloaded, e)
        # Equality of alive references
        self.assertTrue(a == b)
        self.assertFalse(a != b)
        self.assertTrue(a != c)
        self.assertFalse(a == c)
        z = types.int8
        self.assertFalse(a == z)
        self.assertFalse(b == z)
        self.assertFalse(c == z)
        self.assertTrue(a != z)
        self.assertTrue(b != z)
        self.assertTrue(c != z)
        # Hashing and mappings
        s = set([a, b, c])
        self.assertEqual(len(s), 2)
        self.assertIn(a, s)
        self.assertIn(b, s)
        self.assertIn(c, s)
        # Kill the references
        d = e = None
        gc.collect()
        with self.assertRaises(ReferenceError):
            a.overloaded
        with self.assertRaises(ReferenceError):
            b.overloaded
        with self.assertRaises(ReferenceError):
            c.overloaded
        # Dead references are always unequal
        self.assertFalse(a == b)
        self.assertFalse(a == c)
        self.assertFalse(b == c)
        self.assertFalse(a == z)
        self.assertTrue(a != b)
        self.assertTrue(a != c)
        self.assertTrue(b != c)
        self.assertTrue(a != z)

    def test_interning(self):
        # Test interning and lifetime of dynamic types.
        a = types.Dummy('xyzzyx')
        code = a._code
        b = types.Dummy('xyzzyx')
        self.assertIs(b, a)
        wr = weakref.ref(a)
        del a
        gc.collect()
        c = types.Dummy('xyzzyx')
        self.assertIs(c, b)
        # The code is always the same
        self.assertEqual(c._code, code)
        del b, c
        gc.collect()
        self.assertIs(wr(), None)
        d = types.Dummy('xyzzyx')
        # The original code wasn't reused.
        self.assertNotEqual(d._code, code)

    def test_cache_trimming(self):
        # Test that the cache doesn't grow in size when types are
        # created and disposed of.
        gc.collect()
        # Keep strong references to existing types, to avoid spurious failures
        existing_types = [wr() for wr in types._typecache]
        cache_len = len(types._typecache)
        a = types.Dummy('xyzzyx')
        b = types.Dummy('foox')
        self.assertEqual(len(types._typecache), cache_len + 2)
        del a, b
        gc.collect()
        self.assertEqual(len(types._typecache), cache_len)


if __name__ == '__main__':
    unittest.main()
