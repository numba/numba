from __future__ import print_function

import itertools
import random

from numba.compiler import compile_isolated, Flags
from numba import jit, types, utils
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

    def dupsorted_list(self, n, factor=4, offset=10):
        l = (list(range(offset, offset + n // factor)) * (factor + 1))[:n]
        assert len(l) == n
        l.sort()
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

    def fibo(self):
        a = 1
        b = 1
        while True:
            yield a
            a, b = b, a + b

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

    def test_gallop_left(self):
        n = 20
        f = timsort.gallop_left

        def check(l, key, start, stop, hint):
            k = f(key, l, start, stop, hint)
            # Fully check invariants
            self.assertGreaterEqual(k, start)
            self.assertLessEqual(k, stop)
            if k > start:
                self.assertLess(l[k - 1], key)
            if k < stop:
                self.assertGreaterEqual(l[k], key)

        def check_all_hints(l, key, start, stop):
            for hint in range(start, stop):
                check(l, key, start, stop, hint)

        def check_sorted_list(l):
            for key in (l[5], l[15], l[0], -1000, l[-1], 1000):
                check_all_hints(l, key, 0, n)
                check_all_hints(l, key, 1, n - 1)
                check_all_hints(l, key, 8, n - 8)

        l = self.sorted_list(n, offset=100)
        check_sorted_list(l)
        l = self.dupsorted_list(n, offset=100)
        check_sorted_list(l)

    def test_gallop_right(self):
        n = 20
        f = timsort.gallop_right

        def check(l, key, start, stop, hint):
            k = f(key, l, start, stop, hint)
            # Fully check invariants
            self.assertGreaterEqual(k, start)
            self.assertLessEqual(k, stop)
            if k > start:
                self.assertLessEqual(l[k - 1], key)
            if k < stop:
                self.assertGreater(l[k], key)

        def check_all_hints(l, key, start, stop):
            for hint in range(start, stop):
                check(l, key, start, stop, hint)

        def check_sorted_list(l):
            for key in (l[5], l[15], l[0], -1000, l[-1], 1000):
                check_all_hints(l, key, 0, n)
                check_all_hints(l, key, 1, n - 1)
                check_all_hints(l, key, 8, n - 8)

        l = self.sorted_list(n, offset=100)
        check_sorted_list(l)
        l = self.dupsorted_list(n, offset=100)
        check_sorted_list(l)

    def test_merge_compute_minrun(self):
        f = timsort.merge_compute_minrun

        for i in range(0, 64):
            self.assertEqual(f(i), i)
        for i in range(6, 63):
            self.assertEqual(f(2**i), 32)
        for i in self.fibo():
            if i < 64:
                continue
            if i >= 2 ** 63:
                break
            k = f(i)
            self.assertGreaterEqual(k, 32)
            self.assertLessEqual(k, 64)
            if i > 500:
                # i/k is close to, but strictly less than, an exact power of 2
                quot = i // k
                p = 2 ** utils.bit_length(quot)
                self.assertLess(quot, p)
                self.assertGreaterEqual(quot, 0.9 * p)


if __name__ == '__main__':
    unittest.main()
