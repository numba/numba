from __future__ import division

import numpy as np

from numba import unittest_support as unittest
from numba import typeof, types
from numba.compiler import compile_isolated
from .support import TestCase, CompilationCache


def array_flat(arr, out):
    for i, v in enumerate(arr.flat):
        out[i] = v

def array_flat_index(arr, ind):
    return arr.flat[ind]

def array_flat_sum(arr):
    s = 0
    for i, v in enumerate(arr.flat):
        s = s + (i + 1) * v
    return s

def array_ndenumerate_sum(arr):
    s = 0
    for (i, j), v in np.ndenumerate(arr):
        s = s + (i + 1) * (j + 1) * v
    return s

def np_ndindex_empty():
    s = 0
    for ind in np.ndindex(()):
        s += s + len(ind) + 1
    return s

def np_ndindex(x, y):
    s = 0
    n = 0
    for i, j in np.ndindex(x, y):
        s = s + (i + 1) * (j + 1)
    return s

def np_ndindex_array(arr):
    s = 0
    n = 0
    for indices in np.ndindex(arr.shape):
        for i, j in enumerate(indices):
            s = s + (i + 1) * (j + 1)
    return s


class TestArrayIterators(TestCase):
    """
    Test array.flat, np.ndenumerate(), etc.
    """

    def setUp(self):
        self.ccache = CompilationCache()

    def check_array_flat(self, arr, arrty=None):
        out = np.zeros(arr.size, dtype=arr.dtype)
        nb_out = out.copy()
        if arrty is None:
            arrty = typeof(arr)

        cres = compile_isolated(array_flat, [arrty, typeof(out)])
        cfunc = cres.entry_point

        array_flat(arr, out)
        cfunc(arr, nb_out)

        self.assertTrue(np.all(out == nb_out), (out, nb_out))

    def check_array_unary(self, arr, arrty, func):
        cres = compile_isolated(func, [arrty])
        cfunc = cres.entry_point
        self.assertPreciseEqual(cfunc(arr), func(arr))

    def check_array_flat_sum(self, arr, arrty):
        self.check_array_unary(arr, arrty, array_flat_sum)

    def check_array_ndenumerate_sum(self, arr, arrty):
        self.check_array_unary(arr, arrty, array_ndenumerate_sum)

    def test_array_flat_3d(self):
        arr = np.arange(24).reshape(4, 2, 3)

        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 3)
        self.assertEqual(arrty.layout, 'C')
        self.assertTrue(arr.flags.c_contiguous)
        # Test with C-contiguous array
        self.check_array_flat(arr)
        # Test with Fortran-contiguous array
        arr = arr.transpose()
        self.assertFalse(arr.flags.c_contiguous)
        self.assertTrue(arr.flags.f_contiguous)
        self.assertEqual(typeof(arr).layout, 'F')
        self.check_array_flat(arr)
        # Test with non-contiguous array
        arr = arr[::2]
        self.assertFalse(arr.flags.c_contiguous)
        self.assertFalse(arr.flags.f_contiguous)
        self.assertEqual(typeof(arr).layout, 'A')
        self.check_array_flat(arr)

    def test_array_flat_empty(self):
        # Test .flat with various shapes of empty arrays, contiguous
        # and non-contiguous (see issue #846).
        arr = np.zeros(0, dtype=np.int32)
        arr = arr.reshape(0, 2)
        arrty = types.Array(types.int32, 2, layout='C')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        self.check_array_flat_sum(arr, arrty)
        arr = arr.reshape(2, 0)
        arrty = types.Array(types.int32, 2, layout='C')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        self.check_array_flat_sum(arr, arrty)

    def test_array_flat_indexing(self):
        # Test indexing of array.flat object
        pyfunc = array_flat_index
        def check(arr, ind):
            cr = self.ccache.compile(pyfunc, (typeof(arr), typeof(ind)))
            expected = pyfunc(arr, ind)
            self.assertEqual(cr.entry_point(arr, ind), expected)

        arr = np.arange(24).reshape(4, 2, 3)
        for i in range(arr.size):
            check(arr, i)
        arr = arr.T
        for i in range(arr.size):
            check(arr, i)
        arr = arr[::2]
        for i in range(arr.size):
            check(arr, i)
        arr = np.array([42]).reshape(())
        for i in range(arr.size):
            check(arr, i)

    def test_array_ndenumerate_2d(self):
        arr = np.arange(12).reshape(4, 3)
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 2)
        self.assertEqual(arrty.layout, 'C')
        self.assertTrue(arr.flags.c_contiguous)
        # Test with C-contiguous array
        self.check_array_ndenumerate_sum(arr, arrty)
        # Test with Fortran-contiguous array
        arr = arr.transpose()
        self.assertFalse(arr.flags.c_contiguous)
        self.assertTrue(arr.flags.f_contiguous)
        arrty = typeof(arr)
        self.assertEqual(arrty.layout, 'F')
        self.check_array_ndenumerate_sum(arr, arrty)
        # Test with non-contiguous array
        arr = arr[::2]
        self.assertFalse(arr.flags.c_contiguous)
        self.assertFalse(arr.flags.f_contiguous)
        arrty = typeof(arr)
        self.assertEqual(arrty.layout, 'A')
        self.check_array_ndenumerate_sum(arr, arrty)

    def test_array_ndenumerate_empty(self):
        arr = np.zeros(0, dtype=np.int32)
        arr = arr.reshape(0, 2)
        arrty = types.Array(types.int32, 2, layout='C')
        self.check_array_ndenumerate_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        self.check_array_ndenumerate_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        self.check_array_ndenumerate_sum(arr, arrty)
        arr = arr.reshape(2, 0)
        arrty = types.Array(types.int32, 2, layout='C')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='F')
        self.check_array_flat_sum(arr, arrty)
        arrty = types.Array(types.int32, 2, layout='A')
        self.check_array_flat_sum(arr, arrty)

    def test_np_ndindex(self):
        func = np_ndindex
        cres = compile_isolated(func, [types.int32, types.int32])
        cfunc = cres.entry_point
        self.assertPreciseEqual(cfunc(3, 4), func(3, 4))
        self.assertPreciseEqual(cfunc(3, 0), func(3, 0))
        self.assertPreciseEqual(cfunc(0, 3), func(0, 3))
        self.assertPreciseEqual(cfunc(0, 0), func(0, 0))

    def test_np_ndindex_array(self):
        func = np_ndindex_array
        arr = np.arange(12, dtype=np.int32)
        self.check_array_unary(arr, typeof(arr), func)
        arr = arr.reshape((4, 3))
        self.check_array_unary(arr, typeof(arr), func)
        arr = arr.reshape((2, 2, 3))
        self.check_array_unary(arr, typeof(arr), func)

    def test_np_ndindex_empty(self):
        func = np_ndindex_empty
        cres = compile_isolated(func, [])
        cfunc = cres.entry_point
        self.assertPreciseEqual(cfunc(), func())


if __name__ == '__main__':
    unittest.main()
