from __future__ import print_function, absolute_import, division
import numpy as np
from numbapro.testsupport import unittest
from numbapro.cudalib import sorting

SELECT_THRESHOLD = 1e5


class TestRadixSort(unittest.TestCase):
    def _test_sort(self, dtype, counts, reverse=False, seed=0,
                   getindices=False):
        np.random.seed(seed)
        for count in counts:
            data = (np.random.rand(count) * 10 * count).astype(dtype)
            orig = data.copy()
            gold = data.copy()
            gold.sort()

            if reverse:
                gold = gold[::-1]
            rs = sorting.RadixSort(maxcount=count, dtype=data.dtype,
                                   descending=reverse)
            if getindices:
                indices = rs.argsort(data)
            else:
                indices = rs.sort(data)
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

    def test_sort_int32(self):
        counts = [1, 2, 10, 13, 31, 73]
        self._test_sort(np.int32, counts)
        self._test_sort(np.int32, counts, reverse=True)
        self._test_sort(np.int32, counts, reverse=True, getindices=True)

    def test_sort_float64(self):
        counts = [1, 2, 10, 13, 31, 73]
        self._test_sort(np.float64, counts)
        self._test_sort(np.float64, counts, reverse=True)
        self._test_sort(np.float64, counts, reverse=True, getindices=True)

    def _test_select(self, dtype, counts, ks, reverse=False, seed=0,
                     getindices=False):
        np.random.seed(seed)
        for k, count in zip(ks, counts):
            data = (np.random.rand(count) * 10 * count).astype(dtype)
            orig = data.copy()
            gold = data.copy()
            gold.sort()
            if reverse:
                gold = gold[::-1]
            gold = gold[:k]
            rs = sorting.RadixSort(maxcount=count, dtype=data.dtype,
                                   descending=reverse)
            if getindices:
                indices = rs.argselect(keys=data, k=k)
            else:
                indices = rs.select(keys=data, k=k)
            data = data[:k]
            self.assertTrue(np.all(data == gold))
            # print(data, gold)
            if getindices:
                # print(indices)
                # print(orig[indices])
                self.assertTrue(np.all(orig[indices] == gold))
            else:
                self.assertIsNone(indices)

    def test_select_float32(self):
        counts = [1, 2, 10, 13, 31, 73, 100, 101, SELECT_THRESHOLD]
        ks = [1, 1, 3, 5, 10, 60, 99, 101, 1000]
        self._test_select(np.float32, counts, ks)
        self._test_select(np.float32, counts, ks, reverse=True)
        self._test_select(np.float32, counts, ks, reverse=True,
                          getindices=True)

    def test_select_int32(self):
        counts = [1, 2, 10, 13, 31, 73, 100, 101, SELECT_THRESHOLD]
        ks = [1, 1, 3, 5, 10, 60, 99, 101, 1000]
        self._test_select(np.int32, counts, ks)
        self._test_select(np.int32, counts, ks, reverse=True)
        self._test_select(np.int32, counts, ks, reverse=True,
                          getindices=True)

    def test_select_float64(self):
        counts = [1, 2, 10, 13, 31, 73, 100, 101, SELECT_THRESHOLD]
        ks = [1, 1, 3, 5, 10, 60, 99, 101, 1000]
        self._test_select(np.float64, counts, ks)
        self._test_select(np.float64, counts, ks, reverse=True,
                          getindices=True)


class TestSegmentedSort(unittest.TestCase):
    def _test_generic(self, dtype, divby=1):
        keys = np.array(list(reversed(range(100))), dtype=dtype) / divby
        reference = keys.copy()
        original = keys.copy()
        vals = np.arange(keys.size, dtype=np.int32)
        segments = np.array([10, 40, 70], dtype=np.int32)
        sorting.segmented_sort(keys, vals, segments)

        reference[:10].sort()
        reference[10:40].sort()
        reference[40:70].sort()
        reference[70:].sort()

        self.assertTrue(np.all(keys == reference))
        self.assertTrue(np.all(original[vals] == reference))

    def test_float32(self):
        self._test_generic(np.float32, divby=10)

    def test_float64(self):
        self._test_generic(np.float64, divby=10)

    def test_int32(self):
        self._test_generic(np.int32)

    def test_int64(self):
        self._test_generic(np.int64)


if __name__ == '__main__':
    unittest.main()
