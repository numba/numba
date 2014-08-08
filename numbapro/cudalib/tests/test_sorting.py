from __future__ import print_function, absolute_import, division
import numpy as np
from numbapro.testsupport import unittest
from numbapro import cuda
from numbapro.cudalib import sorting


class TestRadixSort(unittest.TestCase):
    def _test_sort(self, dtype, counts, reverse=False, seed=0):
        np.random.seed(seed)
        for count in counts:
            data = np.random.rand(count).astype(dtype)
            orig = data.copy()
            orig.sort()
            if reverse:
                orig = orig[::-1]
            rs = sorting.Radixsort(data.dtype)
            rs.sort(data, reverse=reverse)

            self.assertTrue(np.all(data == orig))

    def test_sort_float32(self):
        counts = [1, 2, 10, 13, 31, 73]
        self._test_sort(np.float32, counts)
        self._test_sort(np.float32, counts, reverse=True)

    def test_sort_float64(self):
        counts = [1, 2, 10, 13, 31, 73]
        self._test_sort(np.float64, counts)
        self._test_sort(np.float64, counts, reverse=True)

    def _test_select(self, dtype, counts, ks, reverse=False, seed=0):
        np.random.seed(seed)
        for k, count in zip(ks, counts):
            data = np.random.rand(count).astype(dtype)
            orig = data.copy()
            orig.sort()
            if reverse:
                orig = orig[::-1]
            orig = orig[:k]
            rs = sorting.Radixsort(data.dtype)
            rs.select(data, k=k, reverse=reverse)
            data = data[:k]
            self.assertTrue(np.all(data == orig))

    def test_select_float32(self):
        counts = [1, 2, 10, 13, 31, 73, 100, 101]
        ks = [1, 1, 3, 5, 10, 60, 99, 101]
        self._test_select(np.float32, counts, ks)
        self._test_select(np.float32, counts, ks, reverse=True)

    def test_select_float64(self):
        counts = [1, 2, 10, 13, 31, 73, 100, 101]
        ks = [1, 1, 3, 5, 10, 60, 99, 101]
        self._test_select(np.float64, counts, ks)
        self._test_select(np.float64, counts, ks, reverse=True)


if __name__ == '__main__':
    unittest.main()
