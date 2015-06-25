"""
Basic tests for the numba.special module.
"""

from __future__ import print_function

import array
import sys

import numpy as np

import numba.unittest_support as unittest
from numba import cffi_support, types
from numba.special import typeof
from .support import TestCase
from .test_numpy_support import ValueTypingTestBase
from .ctypes_usecases import *


class Custom(object):

    @property
    def _numba_type_(self):
        """
        Magic attribute expected by Numba to get the numba type that
        represents this object.
        """
        return types.UniTuple(types.boolean, 42)


class TestTypeof(ValueTypingTestBase, TestCase):
    """
    Test typeof() and, implicitly, typing.Context.get_argument_type().
    """

    def test_number_values(self):
        """
        Test special.typeof() with scalar number values.
        """
        self.check_number_values(typeof)
        # These values mirror Dispatcher semantics
        self.assertEqual(typeof(1), types.int64)
        self.assertEqual(typeof(-1), types.int64)

    def test_datetime_values(self):
        """
        Test special.typeof() with np.timedelta64 values.
        """
        self.check_datetime_values(typeof)

    def test_timedelta_values(self):
        """
        Test special.typeof() with np.timedelta64 values.
        """
        self.check_timedelta_values(typeof)

    def test_array_values(self):
        """
        Test special.typeof() with ndarray values.
        """
        def check(arr, ndim, layout, mutable):
            ty = typeof(arr)
            self.assertIsInstance(ty, types.Array)
            self.assertEqual(ty.ndim, ndim)
            self.assertEqual(ty.layout, layout)
            self.assertEqual(ty.mutable, mutable)

        a1 = np.arange(10)
        check(a1, 1, 'C', True)
        a2 = np.arange(10).reshape(2, 5)
        check(a2, 2, 'C', True)
        check(a2.T, 2, 'F', True)
        a3 = (np.arange(60))[::2].reshape((2, 5, 3))
        check(a3, 3, 'A', True)
        a4 = np.arange(1).reshape(())
        check(a4, 0, 'C', True)
        a4.flags.writeable = False
        check(a4, 0, 'C', False)

    def test_structured_arrays(self):
        def check(arr, dtype, ndim, layout):
            ty = typeof(arr)
            self.assertIsInstance(ty, types.Array)
            self.assertEqual(ty.dtype, dtype)
            self.assertEqual(ty.ndim, ndim)
            self.assertEqual(ty.layout, layout)

        dtype = np.dtype([('m', np.int32), ('n', 'S5')])
        rec_ty = types.Record(id=id(dtype),
                              fields={'m': (types.int32, 0),
                                      'n': (types.CharSeq(5), 4)},
                              size=9,
                              aligned=False,
                              dtype=dtype)

        arr = np.empty(4, dtype=dtype)
        check(arr, rec_ty, 1, "C")
        arr = np.recarray(4, dtype=dtype)
        check(arr, rec_ty, 1, "C")

    @unittest.skipIf(sys.version_info < (2, 7),
                     "buffer protocol not supported on Python 2.6")
    def test_buffers(self):
        if sys.version_info >= (3,):
            b = b"xx"
            ty = typeof(b)
            self.assertEqual(ty, types.Bytes(types.uint8, 1, "C"))
            self.assertFalse(ty.mutable)
            ty = typeof(memoryview(b))
            self.assertEqual(ty, types.MemoryView(types.uint8, 1, "C",
                                                  readonly=True))
            self.assertFalse(ty.mutable)
            ty = typeof(array.array('i', [0, 1, 2]))
            self.assertEqual(ty, types.PyArray(types.int32, 1, "C"))
            self.assertTrue(ty.mutable)

        b = bytearray(10)
        ty = typeof(b)
        self.assertEqual(ty, types.ByteArray(types.uint8, 1, "C"))
        self.assertTrue(ty.mutable)

    def test_none(self):
        ty = typeof(None)
        self.assertEqual(ty, types.none)

    def test_str(self):
        ty = typeof("abc")
        self.assertEqual(ty, types.string)

    def test_tuples(self):
        v = (1, 2)
        self.assertEqual(typeof(v), types.UniTuple(types.int64, 2))
        v = (1, (2.0, 3))
        self.assertEqual(typeof(v),
                         types.Tuple((types.int64,
                                      types.Tuple((types.float64, types.int64))))
                         )

    def test_dtype(self):
        dtype = np.dtype('int64')
        self.assertEqual(typeof(dtype), types.DType(types.int64))

        dtype = np.dtype([('m', np.int32), ('n', 'S5')])
        rec_ty = types.Record(id=id(dtype),
                              fields={'m': (types.int32, 0),
                                      'n': (types.CharSeq(5), 4)},
                              size=9,
                              aligned=False,
                              dtype=dtype)
        self.assertEqual(typeof(dtype), types.DType(rec_ty))

    def test_ctypes(self):
        ty_cos = typeof(c_cos)
        ty_sin = typeof(c_sin)
        self.assertIsInstance(ty_cos, types.ExternalFunctionPointer)
        self.assertEqual(ty_cos.sig.args, (types.float64,))
        self.assertEqual(ty_cos.sig.return_type, types.float64)
        self.assertEqual(ty_cos, ty_sin)
        self.assertNotEqual(ty_cos.get_pointer(c_cos),
                            ty_sin.get_pointer(c_sin))

    @unittest.skipUnless(cffi_support.SUPPORTED, "CFFI not supported")
    def test_cffi(self):
        from .cffi_usecases import cffi_cos, cffi_sin
        ty_cffi_cos = typeof(cffi_cos)
        ty_cffi_sin = typeof(cffi_sin)
        self.assertIsInstance(ty_cffi_cos, types.ExternalFunctionPointer)
        self.assertEqual(ty_cffi_cos.sig.args, (types.float64,))
        self.assertEqual(ty_cffi_cos.sig.return_type, types.float64)
        self.assertEqual(ty_cffi_cos, ty_cffi_sin)
        ty_ctypes_cos = typeof(c_cos)
        self.assertNotEqual(ty_cffi_cos, ty_ctypes_cos)
        self.assertNotEqual(ty_cffi_cos.get_pointer(cffi_cos),
                            ty_cffi_sin.get_pointer(cffi_sin))
        self.assertEqual(ty_cffi_cos.get_pointer(cffi_cos),
                         ty_ctypes_cos.get_pointer(c_cos))

    def test_custom(self):
        ty = typeof(Custom())
        self.assertEqual(ty, types.UniTuple(types.boolean, 42))


if __name__ == '__main__':
    unittest.main()
