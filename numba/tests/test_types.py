"""
Tests for numba.types.
"""

from __future__ import print_function, absolute_import

import gc
try:
    import cPickle as pickle
except ImportError:
    import pickle
import weakref

import numpy as np

from numba.utils import IS_PY3
from numba import abstracttypes, types, typing
from numba import jit, numpy_support
from numba import unittest_support as unittest
from numba.npdatetime import NPDATETIME_SUPPORTED
from .support import TestCase


def gen(x):
    yield x + 1


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

        # Different dtypes
        self.assertNotEqual(types.DType(types.int32), types.DType(types.int64))

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
        if NPDATETIME_SUPPORTED:
            ty = types.NPDatetime('Y')
            self.assertPreciseEqual(ty('1900'), np.datetime64('1900', 'Y'))
            self.assertPreciseEqual(ty('NaT'), np.datetime64('NaT', 'Y'))
            ty = types.NPTimedelta('s')
            self.assertPreciseEqual(ty(5), np.timedelta64(5, 's'))
            self.assertPreciseEqual(ty('NaT'), np.timedelta64('NaT', 's'))
            ty = types.NPTimedelta('')
            self.assertPreciseEqual(ty(5), np.timedelta64(5))
            self.assertPreciseEqual(ty('NaT'), np.timedelta64('NaT'))

    def test_bitwidth_number_types(self):
        """
        All numeric types have bitwidth attribute
        """
        for ty in types.number_domain:
            self.assertTrue(hasattr(ty, "bitwidth"))

    def test_from_bidwidth(self):
        f = types.Integer.from_bitwidth
        self.assertIs(f(32), types.int32)
        self.assertIs(f(8, signed=False), types.uint8)


class TestPickling(TestCase):
    """
    Pickling and unpickling should preserve type identity (singleton-ness)
    and the _code attribute.  This is only a requirement for types that
    can be part of function signatures.
    """

    def predefined_types(self):
        """
        Yield all predefined type instances
        """
        for ty in types.__dict__.values():
            if isinstance(ty, types.Type):
                yield ty

    def check_pickling(self, orig):
        pickled = pickle.dumps(orig, protocol=-1)
        ty = pickle.loads(pickled)
        self.assertIs(ty, orig)
        self.assertGreaterEqual(ty._code, 0)

    def test_predefined_types(self):
        tys = list(self.predefined_types())
        self.assertIn(types.int16, tys)
        for ty in tys:
            self.check_pickling(ty)

    def test_atomic_types(self):
        for unit in ('M', 'ms'):
            ty = types.NPDatetime(unit)
            self.check_pickling(ty)
            ty = types.NPTimedelta(unit)
            self.check_pickling(ty)

    def test_arrays(self):
        for ndim in (0, 1, 2):
            for layout in ('A', 'C', 'F'):
                ty = types.Array(types.int16, ndim, layout)
                self.check_pickling(ty)

    def test_records(self):
        recordtype = np.dtype([('a', np.float64),
                               ('b', np.int32),
                               ('c', np.complex64),
                               ('d', (np.str, 5))])
        ty = numpy_support.from_dtype(recordtype)
        self.check_pickling(ty)
        self.check_pickling(types.Array(ty, 1, 'A'))

    def test_optional(self):
        ty = types.Optional(types.int32)
        self.check_pickling(ty)

    def test_tuples(self):
        ty1 = types.UniTuple(types.int32, 3)
        self.check_pickling(ty1)
        ty2 = types.Tuple((types.int32, ty1))
        self.check_pickling(ty2)

    def test_lists(self):
        ty = types.List(types.int32)
        self.check_pickling(ty)

    def test_generator(self):
        cfunc = jit("(int32,)", nopython=True)(gen)
        sigs = list(cfunc.nopython_signatures)
        ty = sigs[0].return_type
        self.assertIsInstance(ty, types.Generator)
        self.check_pickling(ty)

    # call templates are not picklable
    @unittest.expectedFailure
    def test_external_function_pointers(self):
        from numba.typing import ctypes_utils
        from .ctypes_usecases import c_sin, c_cos
        for fnptr in (c_sin, c_cos):
            ty = ctypes_utils.make_function_type(fnptr)
            self.assertIsInstance(ty, types.ExternalFunctionPointer)
            self.check_pickling(ty)


if __name__ == '__main__':
    unittest.main()
