from __future__ import print_function, absolute_import, division

import numpy as np

from numba import unittest_support as unittest
from numba import njit


class TestDynArray(unittest.TestCase):
    def test_empty_1d(self):
        @njit
        def foo(n):
            arr = np.empty(n)
            for i in range(n):
                arr[i] = i

            return arr

        n = 3
        arr = foo(n)
        np.testing.assert_equal(np.arange(n), arr)
        self.assertEqual(arr.size, n)
        self.assertEqual(arr.shape, (n,))
        self.assertEqual(arr.dtype, np.dtype(np.float64))
        self.assertEqual(arr.strides, (np.dtype(np.float64).itemsize,))
        arr.fill(123)  # test writability
        np.testing.assert_equal(123, arr)
        del arr

    def test_empty_2d(self):
        def pyfunc(m, n):
            arr = np.empty((m, n), np.int32)
            for i in range(m):
                for j in range(n):
                    arr[i, j] = i + j

            return arr

        cfunc = njit(pyfunc)
        m = 4
        n = 3
        expected_arr = pyfunc(m, n)
        got_arr = cfunc(m, n)
        np.testing.assert_equal(expected_arr, got_arr)

        self.assertEqual(expected_arr.size, got_arr.size)
        self.assertEqual(expected_arr.shape, got_arr.shape)
        self.assertEqual(expected_arr.strides, got_arr.strides)

        del got_arr

    def test_return_global_array(self):
        y = np.ones(4, dtype=np.float32)

        def return_external_array():
            return y

        cfunc = njit(return_external_array)
        out = cfunc()

        np.testing.assert_equal(y, out)
        np.testing.assert_equal(y, np.ones(4, dtype=np.float32))
        np.testing.assert_equal(out, np.ones(4, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
