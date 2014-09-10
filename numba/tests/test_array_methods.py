from __future__ import division
from numba import unittest_support as unittest
from numba import typeof
from numba.compiler import compile_isolated
import numpy as np


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


class TestArrayMethods(unittest.TestCase):
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

    def test_array_flat_3d(self):
        arr = np.arange(50).reshape(5, 2, 5)

        arrty = typeof(arr)
        self.assertEqual(arrty.ndim, 3)

        out = np.zeros(arr.size, dtype=arr.dtype)
        nb_out = out.copy()

        cres = compile_isolated(array_flat, [arrty, typeof(out)])
        cfunc = cres.entry_point

        array_flat(arr, out)
        cfunc(arr, nb_out)

        self.assertTrue(np.all(out == nb_out))

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
