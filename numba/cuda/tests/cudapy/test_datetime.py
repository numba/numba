from __future__ import print_function

import numpy as np

from numba import cuda, vectorize, guvectorize, typeof
from numba import unittest_support as unittest
from numba.numpy_support import from_dtype
from numba.tests.support import TestCase
from numba.cuda.testing import SerialMixin


class TestCudaDateTime(SerialMixin, TestCase):
    def test_basic_datetime_kernel(self):
        @cuda.jit
        def foo(start, end, delta):
            for i in range(cuda.grid(1), delta.size, cuda.gridsize(1)):
                delta[i] = end[i] - start[i]

        arr1 = np.arange('2005-02', '2006-02', dtype='datetime64[D]')
        arr2 = arr1 + np.random.randint(0, 10000, arr1.size)
        delta = np.zeros_like(arr1, dtype='timedelta64[D]')

        foo[1, 32](arr1, arr2, delta)

        self.assertPreciseEqual(delta, arr2 - arr1)

    def test_ufunc(self):
        datetime_t = from_dtype(np.dtype('datetime64[D]'))

        @vectorize([(datetime_t, datetime_t)], target='cuda')
        def timediff(start, end):
            return end - start

        arr1 = np.arange('2005-02', '2006-02', dtype='datetime64[D]')
        arr2 = arr1 + np.random.randint(0, 10000, arr1.size)

        delta = timediff(arr1, arr2)

        self.assertPreciseEqual(delta, arr2 - arr1)

    def test_gufunc(self):
        datetime_t = from_dtype(np.dtype('datetime64[D]'))
        timedelta_t = from_dtype(np.dtype('timedelta64[D]'))

        @guvectorize([(datetime_t, datetime_t, timedelta_t[:])], '(),()->()',
                     target='cuda')
        def timediff(start, end, out):
            out[0] = end - start

        arr1 = np.arange('2005-02', '2006-02', dtype='datetime64[D]')
        arr2 = arr1 + np.random.randint(0, 10000, arr1.size)

        delta = timediff(arr1, arr2)

        self.assertPreciseEqual(delta, arr2 - arr1)

if __name__ == '__main__':
    unittest.main()
