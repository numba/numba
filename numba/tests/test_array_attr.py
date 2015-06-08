from __future__ import print_function

import sys

import numpy as np

import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba.numpy_support import from_dtype
from numba import types, njit, typeof
from .support import TestCase


def array_dtype(a):
    return a.dtype


def use_dtype(a, b):
    return a.view(b.dtype)


def array_itemsize(a):
    return a.itemsize


def array_shape(a, i):
    return a.shape[i]


def array_strides(a, i):
    return a.strides[i]


def array_ndim(a):
    return a.ndim


def array_size(a):
    return a.size


def nested_array_itemsize(a):
    return a.f.itemsize


def nested_array_shape(a):
    return a.f.shape


def nested_array_strides(a):
    return a.f.strides


def nested_array_ndim(a):
    return a.f.ndim


def nested_array_size(a):
    return a.f.size


def size_after_slicing_usecase(buf, i):
    sliced = buf[i]
    # Make sure size attribute is not lost
    return sliced.size


def array_ctypes_data(arr):
    return arr.ctypes.data


class TestArrayAttr(TestCase):

    def setUp(self):
        self.a = np.arange(10, dtype=np.int32).reshape(2, 5)

    def get_cfunc(self, pyfunc, argspec):
        cres = compile_isolated(pyfunc, argspec)
        return cres.entry_point

    def test_shape(self):
        pyfunc = array_shape
        cfunc = self.get_cfunc(pyfunc, (types.int32[:,:], types.int32))

        for i in range(self.a.ndim):
            self.assertEqual(pyfunc(self.a, i), cfunc(self.a, i))

    def test_strides(self):
        pyfunc = array_strides
        cfunc = self.get_cfunc(pyfunc, (types.int32[:,:], types.int32))

        for i in range(self.a.ndim):
            self.assertEqual(pyfunc(self.a, i), cfunc(self.a, i))

    def test_ndim(self):
        pyfunc = array_ndim
        cfunc = self.get_cfunc(pyfunc, (types.int32[:,:],))

        self.assertEqual(pyfunc(self.a), cfunc(self.a))

    def test_size(self):
        pyfunc = array_size
        cfunc = self.get_cfunc(pyfunc, (types.int32[:,:],))

        self.assertEqual(pyfunc(self.a), cfunc(self.a))

    def test_itemsize(self):
        pyfunc = array_itemsize
        cfunc = self.get_cfunc(pyfunc, (types.int32[:,:],))

        self.assertEqual(pyfunc(self.a), cfunc(self.a))

    def test_dtype(self):
        pyfunc = array_dtype
        cfunc = self.get_cfunc(pyfunc, (types.int32[:,:],))
        self.assertEqual(pyfunc(self.a), cfunc(self.a))
        dtype = np.dtype([('x', np.int8), ('y', np.int8)])
        arr = np.zeros(4, dtype=dtype)
        cfunc = self.get_cfunc(pyfunc, (typeof(arr),))
        got = cfunc(arr)
        self.assertEqual(cfunc(arr), dtype)

    def test_use_dtype(self):
        # Test using the dtype attribute inside the Numba function itself
        b = np.empty(1, dtype=np.int16)
        pyfunc = use_dtype
        cfunc = self.get_cfunc(pyfunc, (typeof(self.a), typeof(b)))
        expected = pyfunc(self.a, b)
        self.assertPreciseEqual(cfunc(self.a, b), expected)


class TestNestedArrayAttr(unittest.TestCase):
    def setUp(self):
        dtype = np.dtype([('a', np.int32), ('f', np.int32, (2, 5))])
        self.a = np.recarray(1, dtype)[0]
        self.nbrecord = from_dtype(self.a.dtype)

    def get_cfunc(self, pyfunc):
        cres = compile_isolated(pyfunc, (self.nbrecord,))
        return cres.entry_point

    def test_shape(self):
        pyfunc = nested_array_shape
        cfunc = self.get_cfunc(pyfunc)

        self.assertEqual(pyfunc(self.a), cfunc(self.a))

    def test_strides(self):
        pyfunc = nested_array_strides
        cfunc = self.get_cfunc(pyfunc)

        self.assertEqual(pyfunc(self.a), cfunc(self.a))

    def test_ndim(self):
        pyfunc = nested_array_ndim
        cfunc = self.get_cfunc(pyfunc)

        self.assertEqual(pyfunc(self.a), cfunc(self.a))

    def test_size(self):
        pyfunc = nested_array_size
        cfunc = self.get_cfunc(pyfunc)

        self.assertEqual(pyfunc(self.a), cfunc(self.a))

    def test_itemsize(self):
        pyfunc = nested_array_itemsize
        cfunc = self.get_cfunc(pyfunc)

        self.assertEqual(pyfunc(self.a), cfunc(self.a))


class TestSlicedArrayAttr(unittest.TestCase):
    def test_size_after_slicing(self):
        pyfunc = size_after_slicing_usecase
        cfunc = njit(pyfunc)
        arr = np.arange(2 * 5).reshape(2, 5)
        for i in range(arr.shape[0]):
            self.assertEqual(pyfunc(arr, i), cfunc(arr, i))
        arr = np.arange(2 * 5 * 3).reshape(2, 5, 3)
        for i in range(arr.shape[0]):
            self.assertEqual(pyfunc(arr, i), cfunc(arr, i))


class TestArrayCTypes(unittest.TestCase):
    def test_array_ctypes_data(self):
        pyfunc = array_ctypes_data
        cfunc = njit(pyfunc)
        arr = np.arange(3)
        self.assertEqual(pyfunc(arr), cfunc(arr))


if __name__ == '__main__':
    unittest.main()
