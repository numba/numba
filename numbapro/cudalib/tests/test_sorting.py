from __future__ import print_function, absolute_import, division
import numpy as np
from numbapro.testsupport import unittest
from numbapro import cuda
from numbapro.cudalib import sorting


class TestRadixSort(unittest.TestCase):
    def _test_sort(self, dtype, counts, reverse=False, seed=0,
                   getindices=False):
        np.random.seed(seed)
        for count in counts:
            data = np.random.rand(count).astype(dtype)
            orig = data.copy()
            gold = data.copy()
            gold.sort()

            if reverse:
                gold = gold[::-1]
            rs = sorting.Radixsort(data.dtype)
            indices = rs.sort(data, reverse=reverse, getindices=getindices)

            self.assertTrue(np.all(data == gold))
            if getindices:
                self.assertTrue(np.all(orig[indices] == gold))
            else:
                self.assertIsNone(indices)

    def test_sort_float32(self):
        counts = [1, 2, 10, 13, 31, 73]
        self._test_sort(np.float32, counts)
        self._test_sort(np.float32, counts, reverse=True)
        self._test_sort(np.float32, counts, reverse=True, getindices=True)

    def test_sort_float64(self):
        counts = [1, 2, 10, 13, 31, 73]
        self._test_sort(np.float64, counts)
        self._test_sort(np.float64, counts, reverse=True)
        self._test_sort(np.float64, counts, reverse=True, getindices=True)

    def _test_select(self, dtype, counts, ks, reverse=False, seed=0,
                     getindices=False):
        np.random.seed(seed)
        for k, count in zip(ks, counts):
            data = np.random.rand(count).astype(dtype)
            orig = data.copy()
            gold = data.copy()
            gold.sort()
            if reverse:
                gold = gold[::-1]
            gold = gold[:k]
            rs = sorting.Radixsort(data.dtype)
            indices = rs.select(data, k=k, reverse=reverse,
                                getindices=getindices)
            data = data[:k]
            self.assertTrue(np.all(data == gold))
            if getindices:
                self.assertTrue(np.all(orig[indices] == gold))
            else:
                self.assertIsNone(indices)

    def test_select_float32(self):
        counts = [1, 2, 10, 13, 31, 73, 100, 101]
        ks = [1, 1, 3, 5, 10, 60, 99, 101]
        self._test_select(np.float32, counts, ks)
        self._test_select(np.float32, counts, ks, reverse=True)
        self._test_select(np.float32, counts, ks, reverse=True,
                          getindices=True)

    def test_select_float64(self):
        counts = [1, 2, 10, 13, 31, 73, 100, 101]
        ks = [1, 1, 3, 5, 10, 60, 99, 101]
        self._test_select(np.float64, counts, ks)
        self._test_select(np.float64, counts, ks, reverse=True,
                          getindices=True)


if __name__ == '__main__':
    unittest.main()
