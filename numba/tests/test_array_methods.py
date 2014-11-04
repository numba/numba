from __future__ import division

import numpy as np

from numba import unittest_support as unittest
from numba import typeof, types
from numba.compiler import compile_isolated
from .support import TestCase


def array_sum(arr):
    return arr.sum()


def array_sum_global(arr):
    return np.sum(arr)


def array_prod(arr):
    return arr.prod()


def array_prod_global(arr):
    return np.prod(arr)


def array_flat(arr, out):
    for i, v in enumerate(arr.flat):
        out[i] = v

def array_flat_sum(arr):
    s = 0
    for i, v in enumerate(arr.flat):
        s = s + (i + 1) * v
    return s


class TestArrayMethods(TestCase):

    def test_array_sum_int_1d(self):
        arr = np.arange(10, dtype=np.int32)
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 1)
        self.assertEqual(arrty.layout, 'C')

        cres = compile_isolated(array_sum, [arrty])
        cfunc = cres.entry_point

        self.assertEqual(arr.sum(), cfunc(arr))

    def test_array_sum_float_1d(self):
        arr = np.arange(10, dtype=np.float32) / 10
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 1)
        self.assertEqual(arrty.layout, 'C')

        cres = compile_isolated(array_sum, [arrty])
        cfunc = cres.entry_point

        np.testing.assert_allclose(arr.sum(), cfunc(arr), rtol=1e-6)

    def test_array_sum_int_2d(self):
        arr = np.arange(10, dtype=np.int32).reshape(2, 5)
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 2)
        self.assertEqual(arrty.layout, 'C')

        cres = compile_isolated(array_sum, [arrty])
        cfunc = cres.entry_point

        self.assertEqual(arr.sum(), cfunc(arr))

    def test_array_sum_float_2d(self):
        arr = np.arange(10, dtype=np.float32).reshape(2, 5) / 10
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 2)
        self.assertEqual(arrty.layout, 'C')

        cres = compile_isolated(array_sum, [arrty])
        cfunc = cres.entry_point

        np.testing.assert_allclose(arr.sum(), cfunc(arr), rtol=1e-6)

    def test_array_sum_int_3d_any(self):
        arr = (np.arange(60, dtype=np.float32)/10)[::2].reshape((2, 5, 3),
                                                                order='A')
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 3)
        self.assertEqual(arrty.layout, 'A')

        cres = compile_isolated(array_sum, [arrty])
        cfunc = cres.entry_point

        self.assertEqual(arr.sum(), cfunc(arr))

    def test_array_sum_float_3d_any(self):
        arr = (np.arange(60, dtype=np.float32)/10)[::2].reshape((2, 5, 3),
                                                                order='A')
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 3)
        self.assertEqual(arrty.layout, 'A')

        cres = compile_isolated(array_sum, [arrty])
        cfunc = cres.entry_point

        np.testing.assert_allclose(arr.sum(), cfunc(arr))

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

    def check_array_flat_sum(self, arr, arrty):
        cres = compile_isolated(array_flat_sum, [arrty])
        cfunc = cres.entry_point

        self.assertPreciseEqual(cfunc(arr), array_flat_sum(arr))

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
        # Test .flat() with various shapes of empty arrays, contiguous
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

    def test_array_sum_global(self):
        arr = np.arange(10, dtype=np.int32)
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 1)
        self.assertEqual(arrty.layout, 'C')

        cres = compile_isolated(array_sum_global, [arrty])
        cfunc = cres.entry_point

        self.assertEqual(np.sum(arr), cfunc(arr))

    def test_array_prod_int_1d(self):
        arr = np.arange(10, dtype=np.int32) + 1
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 1)
        self.assertEqual(arrty.layout, 'C')

        cres = compile_isolated(array_prod, [arrty])
        cfunc = cres.entry_point

        self.assertEqual(arr.prod(), cfunc(arr))

    def test_array_prod_float_1d(self):
        arr = np.arange(10, dtype=np.float32) + 1 / 10
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 1)
        self.assertEqual(arrty.layout, 'C')

        cres = compile_isolated(array_prod, [arrty])
        cfunc = cres.entry_point

        np.testing.assert_allclose(arr.prod(), cfunc(arr))

    def test_array_prod_global(self):
        arr = np.arange(10, dtype=np.int32)
        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 1)
        self.assertEqual(arrty.layout, 'C')

        cres = compile_isolated(array_prod_global, [arrty])
        cfunc = cres.entry_point

        np.testing.assert_allclose(np.prod(arr), cfunc(arr))


if __name__ == '__main__':
    unittest.main()
