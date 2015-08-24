from __future__ import print_function

import copy
import itertools
import math
import random

import numpy as np

from numba.compiler import compile_isolated, Flags
from numba import jit, types, utils
import numba.unittest_support as unittest
from numba import testing
from .support import TestCase, MemoryLeakMixin

from numba.targets.timsort import (
    make_py_timsort, make_jit_timsort, MergeRun,
    TimsortImplementation)


def make_temp_list(keys, n):
    return [keys[0]] * n

def make_temp_array(keys, n):
    return np.empty(n, keys.dtype)


py_list_timsort = make_py_timsort(make_temp_list)

py_array_timsort = make_py_timsort(make_temp_array)

jit_list_timsort = make_jit_timsort(make_temp_list)

jit_array_timsort = make_jit_timsort(make_temp_array)


def wrap_with_mergestate(timsort, func, _cache={}):
    key = timsort, func
    if key in _cache:
        return _cache[key]

    merge_init = timsort.merge_init

    @timsort.compile
    def wrapper(keys, values, *args):
        ms = merge_init(keys)
        res = func(ms, keys, values, *args)
        return res

    _cache[key] = wrapper
    return wrapper


class BaseTimsortTest(object):

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

    def duprandom_list(self, n, factor=None, offset=10):
        random.seed(42)
        if factor is None:
            factor = int(math.sqrt(n))
        l = (list(range(offset, offset + (n // factor) + 1)) * (factor + 1))[:n]
        assert len(l) == n
        random.shuffle(l)
        return l

    def dupsorted_list(self, n, factor=None, offset=10):
        if factor is None:
            factor = int(math.sqrt(n))
        l = (list(range(offset, offset + (n // factor) + 1)) * (factor + 1))[:n]
        assert len(l) == n, (len(l), n)
        l.sort()
        return l

    def assertSorted(self, orig, result):
        self.assertEqual(len(result), len(orig))
        # sorted() returns a list, so make sure we compare to another list
        self.assertEqual(list(result), sorted(orig))

    def assertSortedValues(self, orig, orig_values, result, result_values):
        self.assertEqual(len(result), len(orig))
        self.assertEqual(list(result), sorted(orig))
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

    def merge_init(self, keys):
        f = self.timsort.merge_init
        return f(keys)

    def test_binarysort(self):
        n = 20
        def check(l, n, start=0):
            res = self.array_factory(l)
            f(res, res, 0, n, start)
            self.assertSorted(l, res)

        f = self.timsort.binarysort
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
            res = self.array_factory(l)
            res_v = self.array_factory(v)
            f(res, res_v, 0, n, start)
            self.assertSortedValues(l, v, res, res_v)

        f = self.timsort.binarysort
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
        f = self.timsort.count_run

        def check(l, lo, hi):
            n, desc = f(self.array_factory(l), lo, hi)
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
        f = self.timsort.gallop_left

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
            l = self.array_factory(l)
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
        f = self.timsort.gallop_right

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
            l = self.array_factory(l)
            for key in (l[5], l[15], l[0], -1000, l[-1], 1000):
                check_all_hints(l, key, 0, n)
                check_all_hints(l, key, 1, n - 1)
                check_all_hints(l, key, 8, n - 8)

        l = self.sorted_list(n, offset=100)
        check_sorted_list(l)
        l = self.dupsorted_list(n, offset=100)
        check_sorted_list(l)

    def test_merge_compute_minrun(self):
        f = self.timsort.merge_compute_minrun

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

    def check_merge_lo_hi(self, func, a, b):
        na = len(a)
        nb = len(b)

        # Add sentinels at start and end, to check they weren't moved
        orig_keys = [42] + a + b + [-42]
        keys = self.array_factory(orig_keys)
        ms = self.merge_init(keys)
        ssa = 1
        ssb = ssa + na

        #new_ms = func(ms, keys, [], ssa, na, ssb, nb)
        new_ms = func(ms, keys, keys, ssa, na, ssb, nb)
        self.assertEqual(keys[0], orig_keys[0])
        self.assertEqual(keys[-1], orig_keys[-1])
        self.assertSorted(orig_keys[1:-1], keys[1:-1])
        # Check the MergeState result
        self.assertGreaterEqual(len(new_ms.keys), len(ms.keys))
        self.assertGreaterEqual(len(new_ms.values), len(ms.values))
        self.assertIs(new_ms.pending, ms.pending)
        self.assertGreaterEqual(new_ms.min_gallop, 1)

    def make_sample_sorted_lists(self, n):
        lists = []
        for offset in (20, 120):
            lists.append(self.sorted_list(n, offset))
            lists.append(self.dupsorted_list(n, offset))
        return lists

    def make_sample_lists(self, n):
        lists = []
        for offset in (20, 120):
            lists.append(self.sorted_list(n, offset))
            lists.append(self.dupsorted_list(n, offset))
            lists.append(self.revsorted_list(n, offset))
            lists.append(self.duprandom_list(n, offset))
        return lists

    def test_merge_lo_hi(self):
        f_lo = self.timsort.merge_lo
        f_hi = self.timsort.merge_hi

        # The larger sizes exercise galloping
        for (na, nb) in [(12, 16), (40, 40), (100, 110), (1000, 1100)]:
            for a, b in itertools.product(self.make_sample_sorted_lists(na),
                                          self.make_sample_sorted_lists(nb)):
                self.check_merge_lo_hi(f_lo, a, b)
                self.check_merge_lo_hi(f_hi, b, a)

    def check_merge_at(self, a, b):
        f = self.timsort.merge_at
        # Prepare the array to be sorted
        na = len(a)
        nb = len(b)
        # Add sentinels at start and end, to check they weren't moved
        orig_keys = [42] + a + b + [-42]
        ssa = 1
        ssb = ssa + na

        stack_sentinels = [MergeRun(-42, -42)] * 2

        def run_merge_at(ms, keys, i):
            #new_ms = f(ms, keys, self.array_factory(()), i)
            new_ms = f(ms, keys, keys, i)
            self.assertEqual(keys[0], orig_keys[0])
            self.assertEqual(keys[-1], orig_keys[-1])
            self.assertSorted(orig_keys[1:-1], keys[1:-1])
            # Check stack state
            self.assertIs(new_ms.pending, ms.pending)
            self.assertEqual(ms.pending[i], (ssa, na + nb))
            self.assertEqual(ms.pending[:i], stack_sentinels)

        # First check with i == len(stack) - 2
        keys = self.array_factory(orig_keys)
        ms = self.merge_init(keys)
        # Push sentinels on stack, to check they weren't touched
        ms.pending.extend(stack_sentinels)
        i = len(ms.pending)
        ms.pending.extend([MergeRun(ssa, na),
                           MergeRun(ssb, nb)])
        run_merge_at(ms, keys, i)
        self.assertEqual(len(ms.pending), i + 1)

        # Now check with i == len(stack) - 3
        keys = self.array_factory(orig_keys)
        ms = self.merge_init(keys)
        # Push sentinels on stack, to check they weren't touched
        ms.pending.extend(stack_sentinels)
        i = len(ms.pending)
        ms.pending.extend([MergeRun(ssa, na),
                           MergeRun(ssb, nb)])
        # A last run (trivial here)
        last_run = MergeRun(ssb + nb, 1)
        ms.pending.append(last_run)
        run_merge_at(ms, keys, i)
        self.assertEqual(len(ms.pending), i + 2)
        self.assertEqual(ms.pending[-1], last_run)

    def test_merge_at(self):
        # The larger sizes exercise galloping
        for (na, nb) in [(12, 16), (40, 40), (100, 110), (500, 510)]:
            for a, b in itertools.product(self.make_sample_sorted_lists(na),
                                          self.make_sample_sorted_lists(nb)):
                self.check_merge_at(a, b)
                self.check_merge_at(b, a)

    def test_merge_force_collapse(self):
        f = self.timsort.merge_force_collapse

        # Test with runs of ascending sizes, then descending sizes
        sizes_list = [(8, 10, 15, 20)]
        sizes_list.append(sizes_list[0][::-1])

        for sizes in sizes_list:
            for chunks in itertools.product(*(self.make_sample_sorted_lists(n)
                                              for n in sizes)):
                # Create runs of the given sizes
                orig_keys = sum(chunks, [])
                keys = self.array_factory(orig_keys)
                ms = self.merge_init(keys)
                pos = 0
                for c in chunks:
                    ms.pending.append(MergeRun(pos, len(c)))
                    pos += len(c)
                # Sanity check
                self.assertEqual(sum(ms.pending[-1]), len(keys))
                # Now merge the runs
                #f(ms, keys, [])
                f(ms, keys, keys)
                # Remaining run is the whole list
                self.assertEqual(ms.pending, [MergeRun(0, len(keys))])
                # The list is now sorted
                self.assertSorted(orig_keys, keys)

    def test_run_timsort(self):
        f = self.timsort.run_timsort

        for size_factor in (1, 10):
            # Make lists to be sorted from three chunks of different kinds.
            sizes = (15, 30, 20)

            all_lists = [self.make_sample_lists(n * size_factor) for n in sizes]
            for chunks in itertools.product(*all_lists):
                orig_keys = sum(chunks, [])
                keys = self.array_factory(orig_keys)
                #print("run_timsort:", keys)
                f(keys)
                #print("-> done")
                # The list is now sorted
                self.assertSorted(orig_keys, keys)

    def test_run_timsort_with_values(self):
        # Run timsort, but also with a values array
        f = self.timsort.run_timsort_with_values

        for size_factor in (1, 5):
            chunk_size = 80 * size_factor
            a = self.dupsorted_list(chunk_size)
            b = self.duprandom_list(chunk_size)
            c = self.revsorted_list(chunk_size)
            orig_keys = a + b + c
            orig_values = list(range(1000, 1000 + len(orig_keys)))

            keys = self.array_factory(orig_keys)
            values = self.array_factory(orig_values)
            f(keys, values)
            # This checks sort stability
            self.assertSortedValues(orig_keys, orig_values, keys, values)


class TestTimsortPurePython(BaseTimsortTest, TestCase):

    timsort = py_list_timsort

    # Much faster than a Numpy array in pure Python
    array_factory = list


class TestTimsortArraysPurePython(BaseTimsortTest, TestCase):

    timsort = py_array_timsort

    def array_factory(self, lst):
        return np.array(lst, dtype=np.int32)


class JITTimsortMixin(object):

    timsort = jit_array_timsort

    test_merge_at = None
    test_merge_force_collapse = None


class TestTimsortArrays(JITTimsortMixin, BaseTimsortTest, TestCase):

    def array_factory(self, lst):
        return np.array(lst, dtype=np.int32)

    def check_merge_lo_hi(self, func, a, b):
        na = len(a)
        nb = len(b)

        func = wrap_with_mergestate(self.timsort, func)

        # Add sentinels at start and end, to check they weren't moved
        orig_keys = [42] + a + b + [-42]
        keys = self.array_factory(orig_keys)
        ssa = 1
        ssb = ssa + na

        new_ms = func(keys, keys, ssa, na, ssb, nb)
        self.assertEqual(keys[0], orig_keys[0])
        self.assertEqual(keys[-1], orig_keys[-1])
        self.assertSorted(orig_keys[1:-1], keys[1:-1])


if __name__ == '__main__':
    unittest.main()
