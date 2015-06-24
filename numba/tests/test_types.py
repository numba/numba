"""
Tests for numba.types.
"""

from __future__ import print_function, absolute_import

import gc
import weakref

from numba.utils import IS_PY3
from numba import abstracttypes, types, typing
from numba import unittest_support as unittest
from .support import TestCase


class Dummy(object):
    pass


class TestTypes(TestCase):

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
        # Same arguments but different return types
        get_pointer = None
        sig_a = typing.signature(types.intp, types.intp)
        sig_b = typing.signature(types.voidptr, types.intp)
        a = types.ExternalFunctionPointer(sig=sig_a, get_pointer=get_pointer)
        b = types.ExternalFunctionPointer(sig=sig_b, get_pointer=get_pointer)
        self.assertNotEqual(a, b)
        # Different call convention
        a = types.ExternalFunctionPointer(sig=sig_a, get_pointer=get_pointer)
        b = types.ExternalFunctionPointer(sig=sig_a, get_pointer=get_pointer,
                                          cconv='stdcall')
        self.assertNotEqual(a, b)
        # Different get_pointer
        a = types.ExternalFunctionPointer(sig=sig_a, get_pointer=get_pointer)
        b = types.ExternalFunctionPointer(sig=sig_a, get_pointer=object())
        self.assertNotEqual(a, b)

        # Different template classes bearing the same name
        class DummyTemplate(object):
            key = "foo"
        a = types.BoundFunction(DummyTemplate, types.int32)
        class DummyTemplate(object):
            key = "bar"
        b = types.BoundFunction(DummyTemplate, types.int32)
        self.assertNotEqual(a, b)

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
        cache = abstracttypes._typecache
        gc.collect()
        # Keep strong references to existing types, to avoid spurious failures
        existing_types = [wr() for wr in cache]
        cache_len = len(cache)
        a = types.Dummy('xyzzyx')
        b = types.Dummy('foox')
        self.assertEqual(len(cache), cache_len + 2)
        del a, b
        gc.collect()
        self.assertEqual(len(cache), cache_len)

    def test_array_notation(self):
        def check(arrty, scalar, ndim, layout):
            self.assertIs(arrty.dtype, scalar)
            self.assertEqual(arrty.ndim, ndim)
            self.assertEqual(arrty.layout, layout)
        scalar = types.int32
        check(scalar[:], scalar, 1, 'A')
        check(scalar[::1], scalar, 1, 'C')
        check(scalar[:,:], scalar, 2, 'A')
        check(scalar[:,::1], scalar, 2, 'C')
        check(scalar[::1,:], scalar, 2, 'F')

    def test_call_notation(self):
        # Function call signature
        i = types.int32
        d = types.double
        self.assertEqual(i(), typing.signature(i))
        self.assertEqual(i(d), typing.signature(i, d))
        self.assertEqual(i(d, d), typing.signature(i, d, d))
        # Value cast
        self.assertPreciseEqual(i(42.5), 42)
        self.assertPreciseEqual(d(-5), -5.0)

    def test_bitwidth_number_types(self):
        """
        All numeric types have bitwidth attribute
        """
        for ty in types.number_domain:
            self.assertTrue(hasattr(ty, "bitwidth"))


if __name__ == '__main__':
    unittest.main()
