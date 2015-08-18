from __future__ import print_function

import itertools
import random

from numba.compiler import compile_isolated, Flags
from numba import jit, types
import numba.unittest_support as unittest
from numba import testing
from .support import TestCase, MemoryLeakMixin

from numba.targets import timsort


class TestTimsortPurePython(TestCase):

    def random_list(self, n, offset=10):
        random.seed(42)
        l = list(range(offset, offset + n))
        random.shuffle(l)
        return l

    def sorted_list(self, n, offset=10):
        return list(range(offset, offset + n))

    def revsorted_list(self, n, offset=10):
        return list(range(offset, offset + n))[::-1]

    def initially_sorted_list(self, n, m=None, offset=10):
        if m is None:
            m = n // 2
        l = self.sorted_list(m, offset)
        l += self.random_list(n - m, offset=l[-1] + offset)
        return l

    def duprandom_list(self, n, factor=4, offset=10):
        random.seed(42)
        l = (list(range(offset, offset + n // factor)) * (factor + 1))[:n]
        assert len(l) == n
        random.shuffle(l)
        return l

    def assertSorted(self, orig, result):
        self.assertEqual(len(result), len(orig))
        self.assertEqual(result, sorted(orig))

    def assertSortedValues(self, orig, orig_values, result, result_values):
        self.assertEqual(len(result), len(orig))
        self.assertEqual(result, sorted(orig))
        zip_sorted = sorted(zip(orig, orig_values), key=lambda x: x[0])
        zip_result = list(zip(result, result_values))
        self.assertEqual(zip_sorted, zip_result)
        # Check stability
        for i in range(len(zip_result) - 1):
            (k1, v1), (k2, v2) = zip_result[i], zip_result[i + 1]
            if k1 == k2:
                # Assuming values are unique, which is enforced by the tests
                self.assertLess(orig_values.index(v1), orig_values.index(v2))

    def test_binarysort(self):
        n = 20
        def check(l, n, start=0):
            res = l[:]
            f(res, (), 0, n, start)
            self.assertSorted(l, res)

        f = timsort.binarysort
        l = self.sorted_list(n)
        check(l, n)
        check(l, n, n//2)
        l = self.revsorted_list(n)
        check(l, n)
        l = self.initially_sorted_list(n, n//2)
        check(l, n)
        check(l, n, n//2)
        l = self.revsorted_list(n)
        check(l, n)
        l = self.random_list(n)
        check(l, n)
        l = self.duprandom_list(n)
        check(l, n)

    def test_binarysort_with_values(self):
        n = 20
        v = list(range(100, 100+n))

        def check(l, n, start=0):
            res = l[:]
            res_v = v[:]
            f(res, res_v, 0, n, start)
            self.assertSortedValues(l, v, res, res_v)

        f = timsort.binarysort
        l = self.sorted_list(n)
        check(l, n)
        check(l, n, n//2)
        l = self.revsorted_list(n)
        check(l, n)
        l = self.initially_sorted_list(n, n//2)
        check(l, n)
        check(l, n, n//2)
        l = self.revsorted_list(n)
        check(l, n)
        l = self.random_list(n)
        check(l, n)
        l = self.duprandom_list(n)
        check(l, n)

    def test_count_run(self):
        n = 16
        f = timsort.count_run

        def check(l, lo, hi):
            n, desc = f(l, lo, hi)
            # Fully check invariants
            if desc:
                for k in range(lo, lo + n - 1):
                    a, b = l[k], l[k + 1]
                    self.assertGreater(a, b)
                if lo + n < hi:
                    self.assertLessEqual(l[lo + n - 1], l[lo + n])
            else:
                for k in range(lo, lo + n - 1):
                    a, b = l[k], l[k + 1]
                    self.assertLessEqual(a, b)
                if lo + n < hi:
                    self.assertGreater(l[lo + n - 1], l[lo + n], l)


        l = self.sorted_list(n, offset=100)
        check(l, 0, n)
        check(l, 1, n - 1)
        check(l, 1, 2)
        l = self.revsorted_list(n, offset=100)
        check(l, 0, n)
        check(l, 1, n - 1)
        check(l, 1, 2)
        l = self.random_list(n, offset=100)
        for i in range(len(l) - 1):
            check(l, i, n)
        l = self.duprandom_list(n, offset=100)
        for i in range(len(l) - 1):
            check(l, i, n)


if __name__ == '__main__':
    unittest.main()
