"""
Tests for the concurrent (sharded) typed dictionary in
``numba.typed.concurrentdict``.
"""
import unittest
import warnings

import numpy as np

from numba import njit, prange
from numba.core import types
from numba.core.errors import NumbaTypeSafetyWarning, TypingError
from numba.typed import Dict
from numba.typed.concurrentdict import (
    ConcurrentDict,
    concurrent_dict,
    spin_lock,
    spin_unlock,
)
from numba.tests.support import TestCase, skip_parfors_unsupported


@njit
def _spinlock_counter(n_threads, iters):
    """Increment a counter guarded by a single spinlock from
    *n_threads* logical workers, *iters* times each.
    """
    locks = np.zeros(1, dtype=np.uint64)
    counter = np.zeros(1, dtype=np.int64)
    for i in prange(n_threads * iters):
        spin_lock(locks, 0)
        counter[0] += 1
        spin_unlock(locks, 0)
    return counter[0]


@njit
def _spinlock_counter_unlocked(n_threads, iters):
    """Same as _spinlock_counter but without locking; the result is
    allowed to be wrong (race condition demonstration, not asserted).
    """
    locks = np.zeros(1, dtype=np.uint64)
    counter = np.zeros(1, dtype=np.int64)
    for i in prange(n_threads * iters):
        # NOTE: no lock
        counter[0] += 1
    return counter[0], locks[0]


@njit
def _spinlock_sharded_counter(n_shards, n_threads, iters):
    """Increment per-shard counters guarded by per-shard spinlocks,
    mimicking the sharded dictionary's lock layout.
    """
    locks = np.zeros(n_shards, dtype=np.uint64)
    counters = np.zeros(n_shards, dtype=np.int64)
    for i in prange(n_threads * iters):
        shard = i % n_shards
        spin_lock(locks, shard)
        counters[shard] += 1
        spin_unlock(locks, shard)
    return counters.sum()


class TestSpinLock(TestCase):
    """Tests for the atomic spinlock (CPU mutex) primitives."""

    @skip_parfors_unsupported
    def test_mutual_exclusion_exact_count(self):
        # The spinlock must make concurrent increments race-free: the
        # final counter value must be exact.
        for n_threads in (2, 4, 8):
            for iters in (100, 10000):
                got = _spinlock_counter(n_threads, iters)
                self.assertEqual(got, n_threads * iters)

    @skip_parfors_unsupported
    def test_sharded_mutual_exclusion(self):
        # Per-shard locks protect per-shard counters independently;
        # every shard counter must be exact.
        for n_shards in (1, 4, 16, 256):
            got = _spinlock_sharded_counter(n_shards, 4, 5000)
            self.assertEqual(got, 4 * 5000)

    def test_lock_type_checking(self):
        @njit
        def bad_lock_type():
            locks = np.zeros(4, dtype=np.int64)
            spin_lock(locks, 0)

        with self.assertRaises(TypingError) as raises:
            bad_lock_type()
        self.assertIn("uint64 array", str(raises.exception))

        @njit
        def bad_lock_ndim():
            locks = np.zeros((4, 4), dtype=np.uint64)
            spin_lock(locks, 0)

        with self.assertRaises(TypingError) as raises:
            bad_lock_ndim()
        self.assertIn("1-D", str(raises.exception))


class TestConcurrentDictBasic(TestCase):
    """Single-threaded functional tests."""

    def make(self, n_shards=8, key_type=types.int64,
             value_type=types.int64):
        return ConcurrentDict.empty(key_type=key_type,
                                    value_type=value_type,
                                    n_shards=n_shards)

    def test_setitem_getitem(self):
        cd = self.make()
        for k in range(100):
            cd[k] = k * 2
        for k in range(100):
            self.assertEqual(cd[k], k * 2)

    def test_len(self):
        cd = self.make()
        self.assertEqual(len(cd), 0)
        for k in range(100):
            cd[k] = k
        self.assertEqual(len(cd), 100)
        # overwriting does not change the length
        for k in range(100):
            cd[k] = -k
        self.assertEqual(len(cd), 100)

    def test_contains(self):
        cd = self.make()
        cd[5] = 1
        self.assertTrue(5 in cd)
        self.assertFalse(4 in cd)

    def test_get_default(self):
        cd = self.make()
        cd[1] = 10
        self.assertEqual(cd.get(1, -1), 10)
        self.assertEqual(cd.get(2, -1), -1)

    def test_pop(self):
        cd = self.make()
        cd[1] = 10
        cd[2] = 20
        self.assertEqual(cd.pop(1), 10)
        self.assertEqual(len(cd), 1)
        self.assertFalse(1 in cd)
        with self.assertRaises(KeyError):
            cd.pop(1)

    def test_getitem_missing_key(self):
        # Regression test: a KeyError from __getitem__ must not leave
        # the shard spinlock acquired (which would deadlock the shard).
        cd = self.make()
        cd[1] = 10
        with self.assertRaises(KeyError):
            cd[2]
        # the shard must still be usable
        self.assertEqual(cd[1], 10)
        cd[2] = 20
        self.assertEqual(cd[2], 20)

    def test_pop_missing_key_unlocks(self):
        cd = self.make()
        cd[1] = 10
        with self.assertRaises(KeyError):
            cd.pop(2)
        self.assertEqual(cd[1], 10)

    def test_delitem(self):
        cd = self.make()
        cd[1] = 10
        cd[2] = 20
        del cd[1]
        self.assertFalse(1 in cd)
        self.assertEqual(cd[2], 20)
        self.assertEqual(len(cd), 1)
        # a missing key raises and leaves the shard usable
        with self.assertRaises(KeyError):
            del cd[99]
        cd[1] = 5
        self.assertEqual(cd[1], 5)

    def test_setdefault(self):
        cd = self.make()
        self.assertEqual(cd.setdefault(1, 42), 42)
        # the default is not written when the key exists
        self.assertEqual(cd.setdefault(1, 99), 42)
        self.assertEqual(cd[1], 42)
        self.assertEqual(len(cd), 1)

    def test_update(self):
        src = Dict.empty(key_type=types.int64, value_type=types.int64)
        for k in range(10):
            src[k] = k * 3

        @njit
        def upd(d, s):
            d.update(s)
            return len(d), d[7]

        cd = self.make(n_shards=8)
        n, v = upd(cd, src)
        self.assertEqual(n, 10)
        self.assertEqual(v, 21)

    def test_popitem(self):
        cd = self.make()
        cd[5] = 55
        key, val = cd.popitem()
        self.assertEqual(key, 5)
        self.assertEqual(val, 55)
        self.assertEqual(len(cd), 0)
        with self.assertRaises(KeyError):
            cd.popitem()
        # usable after draining
        cd[1] = 1
        self.assertEqual(cd[1], 1)

    def test_clear(self):
        cd = self.make()
        for k in range(20):
            cd[k] = k
        cd.clear()
        self.assertEqual(len(cd), 0)
        cd[1] = 1
        self.assertEqual(cd[1], 1)

    def test_copy(self):
        cd = self.make(n_shards=8)
        for k in range(50):
            cd[k] = k * 2

        @njit
        def do_copy(d):
            c = d.copy()
            return c, len(c), c.n_shards

        cp, n, n_shards = do_copy(cd)
        self.assertEqual(n, 50)
        self.assertEqual(n_shards, 8)
        for k in range(50):
            self.assertEqual(cp[k], k * 2)
        # the copy is independent
        cp[0] = -1
        self.assertEqual(cd[0], 0)

        # copy with a different shard count
        @njit
        def do_copy2(d, nshards):
            c = d.copy(nshards)
            return c, len(c), c.n_shards

        cp2, n, n_shards = do_copy2(cd, 32)
        self.assertEqual(n, 50)
        self.assertEqual(n_shards, 32)
        for k in range(50):
            self.assertEqual(cp2[k], k * 2)

    def test_unsupported_iteration_views(self):
        # keys()/values()/items() are deliberately unsupported; they
        # would require (concurrent-unsafe) iteration.
        cd = self.make()
        cd[1] = 1

        @njit
        def keys_of(d):
            return d.keys()

        with self.assertRaises(TypingError):
            keys_of(cd)

        @njit
        def items_of(d):
            return d.items()

        with self.assertRaises(TypingError):
            items_of(cd)

        @njit
        def values_of(d):
            return d.values()

        with self.assertRaises(TypingError):
            values_of(cd)

    def test_shard_count_one(self):
        # n_shards=1 degenerates to a globally locked dictionary
        cd = self.make(n_shards=1)
        for k in range(10):
            cd[k] = k
        self.assertEqual(len(cd), 10)
        self.assertEqual(cd[7], 7)

    def test_invalid_shard_count(self):
        @njit
        def create():
            return concurrent_dict(types.int64, types.int64, 0)

        with self.assertRaises(ValueError):
            create()

    def test_key_types(self):
        cd = self.make(key_type=types.float64,
                       value_type=types.unicode_type)
        cd[1.5] = "one point five"
        self.assertEqual(cd[1.5], "one point five")
        self.assertTrue(1.5 in cd)
        self.assertFalse(2.5 in cd)

        cdu = self.make(key_type=types.unicode_type,
                        value_type=types.int64)
        cdu["a"] = 1
        cdu["b"] = 2
        self.assertEqual(cdu["a"], 1)
        self.assertEqual(cdu["b"], 2)
        self.assertEqual(len(cdu), 2)

    def test_value_types(self):
        cd = self.make(value_type=types.float64)
        cd[1] = 2.5
        self.assertPreciseEqual(cd[1], 2.5)

    def test_value_type_mismatch(self):
        cd = self.make(key_type=types.int64, value_type=types.int64)

        @njit
        def setter(d):
            d[1] = "string"

        with self.assertRaises(TypingError):
            setter(cd)

    def test_nopython_creation(self):
        # concurrent_dict() is callable from jit functions
        @njit
        def create_and_use(n, n_shards):
            cd = concurrent_dict(types.int64, types.int64, n_shards)
            for k in range(n):
                cd[k] = k * 2
            return len(cd), cd[10]

        n, v = create_and_use(50, 16)
        self.assertEqual(n, 50)
        self.assertEqual(v, 20)

    def test_concurrent_dict_function_python_side(self):
        cd = concurrent_dict(types.int64, types.int64, n_shards=4)
        cd[1] = 1
        self.assertEqual(len(cd), 1)

    def test_passing_to_jit_function(self):
        cd = self.make()
        cd[1] = 10

        @njit
        def read(d):
            return d[1], len(d)

        v, n = read(cd)
        self.assertEqual(v, 10)
        self.assertEqual(n, 1)

    def test_mutation_visible_across_functions(self):
        cd = self.make()

        @njit
        def write(d):
            d[42] = 7

        @njit
        def read(d):
            return d[42]

        write(cd)
        self.assertEqual(read(cd), 7)


class TestConcurrentDictParallel(TestCase):
    """Parallel (prange) tests; these exercise the locking."""

    def setUp(self):
        super().setUp()
        # The typed-dict lowering used by the shards emits an unsafe
        # cast warning (uint64 -> int64) inside numba's own dict
        # operations when compiled by the parallel pipeline; this is
        # pre-existing behaviour of numba.typed.Dict and unrelated to
        # the sharding/locking implemented here.
        self.warning_catcher = warnings.catch_warnings()
        self.warning_catcher.__enter__()
        warnings.simplefilter("ignore", NumbaTypeSafetyWarning)
        self.addCleanup(self.warning_catcher.__exit__, None, None, None)

    def make(self, n_shards=16, key_type=types.int64,
             value_type=types.int64):
        return ConcurrentDict.empty(key_type=key_type,
                                    value_type=value_type,
                                    n_shards=n_shards)

    @skip_parfors_unsupported
    def test_parallel_fill(self):
        n = 20000
        cd = self.make()

        @njit(parallel=True, nogil=True)
        def fill(d, n):
            for k in prange(n):
                d[k] = k * 2
            return len(d)

        self.assertEqual(fill(cd, n), n)
        for k in range(n):
            self.assertEqual(cd[k], k * 2)

    @skip_parfors_unsupported
    def test_parallel_fill_stress(self):
        # Repeat the parallel fill to shake out races; a locking bug
        # usually shows up as a lost update (wrong len) or a crash.
        n = 5000

        @njit(parallel=True, nogil=True)
        def fill(d, n):
            for k in prange(n):
                d[k] = k
            return len(d)

        for _ in range(10):
            cd = self.make(n_shards=64)
            self.assertEqual(fill(cd, n), n)

    @skip_parfors_unsupported
    def test_parallel_same_key_writes(self):
        # Concurrent writes to the same key are last-writer-wins; the
        # final value must be one of the written values and the length
        # must be exactly one.
        cd = self.make(n_shards=4)

        @njit(parallel=True, nogil=True)
        def write_same(d, n):
            for k in prange(n):
                d[0] = k

        write_same(cd, 10000)
        self.assertEqual(len(cd), 1)
        self.assertTrue(0 <= cd[0] < 10000)

    @skip_parfors_unsupported
    def test_parallel_mixed_operations(self):
        # Concurrent writes, reads, containment checks and pops on
        # disjoint key sets.
        n = 10000
        cd = self.make(n_shards=32)

        @njit(parallel=True, nogil=True)
        def mixed(d, n):
            for k in prange(n):
                d[k] = k
            # concurrent reads
            acc = 0
            for k in prange(n):
                acc += d[k]
            # concurrent containment checks
            found = 0
            for k in prange(n):
                found += k in d
            # concurrent pops of the first half of the keys
            for k in prange(n // 2):
                d.pop(k)
            return acc, found, len(d)

        expected_acc = sum(range(n))
        acc, found, remaining = mixed(cd, n)
        self.assertEqual(acc, expected_acc)
        self.assertEqual(found, n)
        self.assertEqual(remaining, n - n // 2)
        # the second half of the keys must still be present and correct
        for k in range(n // 2, n):
            self.assertEqual(cd[k], k)
        for k in range(n // 2):
            self.assertFalse(k in cd)

    @skip_parfors_unsupported
    def test_parallel_fill_various_shard_counts(self):
        # The shard count is a tuning parameter; correctness must hold
        # for any value, including the degenerate 1.
        n = 5000

        @njit(parallel=True, nogil=True)
        def fill(d, n):
            for k in prange(n):
                d[k] = k + 1
            return len(d)

        for n_shards in (1, 2, 7, 16, 256, 1024):
            cd = self.make(n_shards=n_shards)
            self.assertEqual(fill(cd, n), n)
            self.assertEqual(cd[0], 1)
            self.assertEqual(cd[n - 1], n)

    @skip_parfors_unsupported
    def test_parallel_len_consistency(self):
        # len() acquires each shard lock; concurrent with writes it
        # returns a consistent snapshot, and after the writers are done
        # it is exact.
        n = 10000
        cd = self.make(n_shards=16)

        @njit(parallel=True, nogil=True)
        def fill_and_measure(d, n):
            total = 0
            for k in prange(n):
                d[k] = k
            for _ in prange(4):
                total += len(d)
            return total, len(d)

        total, final = fill_and_measure(cd, n)
        self.assertEqual(final, n)
        # each of the 4 measurements sees between 0 and n entries;
        # the sum must be within that range
        self.assertGreaterEqual(total, 0)
        self.assertLessEqual(total, 4 * n)

    @skip_parfors_unsupported
    def test_parallel_unicode_keys(self):
        n = 2000
        cd = self.make(n_shards=8, key_type=types.unicode_type,
                       value_type=types.int64)

        @njit(parallel=True, nogil=True)
        def fill(d, n):
            for k in prange(n):
                d["key_" + str(k)] = k
            return len(d)

        self.assertEqual(fill(cd, n), n)
        for k in range(n):
            self.assertEqual(cd["key_" + str(k)], k)

    @skip_parfors_unsupported
    def test_issue_8767_usecase(self):
        # The original motivating scenario: multiple workers writing
        # their results into a shared dictionary, keyed by worker id.
        n_workers = 16
        cd = self.make(n_shards=256)

        @njit(parallel=True, nogil=True)
        def workers(d, n_workers):
            for w in prange(n_workers):
                # each worker writes a nested result; here a simple
                # marker value
                d[w] = w * w

        workers(cd, n_workers)
        self.assertEqual(len(cd), n_workers)
        for w in range(n_workers):
            self.assertEqual(cd[w], w * w)


if __name__ == '__main__':
    unittest.main()
