# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import numpy as np

from numba import (njit, set_num_threads, get_num_threads, get_thread_num,
                   prange, config, threading_layer)
from numba import unittest_support as unittest
from .support import TestCase, skip_parfors_unsupported

class TestNumThreads(TestCase):
    _numba_parallel_test_ = False

    def setUp(self):
        # Make sure the num_threads is set to the max. This also makes sure
        # the threads are launched.
        set_num_threads(config.NUMBA_NUM_THREADS)

    @skip_parfors_unsupported
    @unittest.skipIf(config.NUMBA_NUM_THREADS < 2, "Not enough CPU cores")
    def test_set_num_threads_basic(self):
        max_threads = config.NUMBA_NUM_THREADS

        self.assertEqual(get_num_threads(), max_threads)
        set_num_threads(2)
        self.assertEqual(get_num_threads(), 2)
        set_num_threads(max_threads)
        self.assertEqual(get_num_threads(), max_threads)

        with self.assertRaises(ValueError):
            set_num_threads(0)

        with self.assertRaises(ValueError):
            set_num_threads(max_threads + 1)

    @skip_parfors_unsupported
    @unittest.skipIf(config.NUMBA_NUM_THREADS < 2, "Not enough CPU cores")
    def test_set_num_threads_basic_jit(self):
        max_threads = config.NUMBA_NUM_THREADS

        @njit
        def get_n():
            return get_num_threads()

        @njit
        def set_n(n):
            set_num_threads(n)

        self.assertEqual(get_n(), max_threads)
        set_n(2)
        self.assertEqual(get_n(), 2)
        set_n(max_threads)
        self.assertEqual(get_n(), max_threads)

        @njit
        def set_get_n(n):
            set_num_threads(n)
            return get_num_threads()

        self.assertEqual(set_get_n(2), 2)
        self.assertEqual(set_get_n(max_threads), max_threads)

        with self.assertRaises(ValueError):
            set_n(0)

        with self.assertRaises(ValueError):
            set_n(max_threads + 1)

    @skip_parfors_unsupported
    @unittest.skipIf(config.NUMBA_NUM_THREADS < 2, "Not enough CPU cores")
    def test_set_num_threads_outside_jit(self):

        # Test set_num_threads outside a jitted function
        set_num_threads(2)

        @njit(parallel=True)
        def test_func():
            x = 5
            buf = np.empty((x,))
            for i in prange(x):
                buf[i] = get_num_threads()
            return buf

        out = test_func()
        np.testing.assert_equal(out, 2)

    @skip_parfors_unsupported
    @unittest.skipIf(config.NUMBA_NUM_THREADS < 2, "Not enough CPU cores")
    def test_set_num_threads_inside_jit(self):
        # Test set_num_threads inside a jitted function
        @njit(parallel=True)
        def test_func(nthreads):
            x = 5
            buf = np.empty((x,))
            set_num_threads(nthreads)
            for i in prange(x):
                buf[i] = get_num_threads()
            return buf

        mask = 2
        out = test_func(mask)
        np.testing.assert_equal(out, mask)

    @skip_parfors_unsupported
    @unittest.skipIf(config.NUMBA_NUM_THREADS < 2, "Not enough CPU cores")
    def test_get_num_threads_truth_outside_jit(self):

        for mask in range(2, min(6, config.NUMBA_NUM_THREADS + 1)):
            set_num_threads(mask)

            # a lot of work, hopefully will trigger "mask" count of threads to
            # join the parallel region (for those backends with dynamic threads)
            @njit(parallel=True)
            def test_func():
                x = 5000000
                buf = np.empty((x,))
                for i in prange(x):
                    buf[i] = get_thread_num()
                return len(np.unique(buf)), get_num_threads()

            out = test_func()
            self.assertEqual(out, (mask, mask))

    @skip_parfors_unsupported
    @unittest.skipIf(config.NUMBA_NUM_THREADS < 2, "Not enough CPU cores")
    def test_get_num_threads_truth_inside_jit(self):

        for mask in range(2, min(6, config.NUMBA_NUM_THREADS + 1)):

            # a lot of work, hopefully will trigger "mask" count of threads to
            # join the parallel region (for those backends with dynamic threads)
            @njit(parallel=True)
            def test_func():
                set_num_threads(mask)
                x = 5000000
                buf = np.empty((x,))
                for i in prange(x):
                    buf[i] = get_thread_num()
                return len(np.unique(buf)), get_num_threads()

            out = test_func()
            self.assertEqual(out, (mask, mask))

    # this test can only run on OpenMP (providing OMP_MAX_ACTIVE_LEVELS is not
    # set or >= 2) and TBB backends
    @skip_parfors_unsupported
    @unittest.skipIf(config.NUMBA_NUM_THREADS < 2, "Not enough CPU cores")
    def test_nested_parallelism_1(self):
        if threading_layer() == 'workqueue':
            self.skipTest("workqueue is not threadsafe")

        # check that get_thread_num is ok in nesting
        mask = config.NUMBA_NUM_THREADS - 1

        N = 4
        M = 8

        def gen(fid):
            @njit(parallel=True)
            def child_func(buf):
                M, N = buf.shape
                for i in prange(N):
                    buf[fid, i] = get_num_threads()
            return child_func

        child1 = gen(1)
        child2 = gen(2)
        child3 = gen(3)

        @njit(parallel=True)
        def test_func(nthreads):
            acc = 0
            buf = np.zeros((M, N))
            set_num_threads(nthreads)
            for i in prange(M):
                local_mask = 1 + i % mask
                set_num_threads(local_mask)  # set threads in parent function
                if local_mask == 1:
                    child1(buf)
                elif local_mask == 2:
                    child2(buf)
                elif local_mask == 3:
                    child3(buf)
                acc += get_num_threads()
            return acc, buf

        got_acc, got_arr = test_func(mask)
        exp_acc, exp_arr = test_func.py_func(mask)
        self.assertEqual(exp_acc, got_acc)
        np.testing.assert_equal(exp_arr, got_arr)

        # check the maths reconciles
        math_acc = np.sum(1 + np.arange(M) % mask)
        self.assertEqual(math_acc, got_acc)
        math_arr = np.zeros((M, N))
        for i in range(1, 4):  # there's branches on 1, 2, 3
            math_arr[i, :] = i
        np.testing.assert_equal(math_arr, got_arr)

    # this test can only run on OpenMP (providing OMP_MAX_ACTIVE_LEVELS is not
    # set or >= 2) and TBB backends
    @skip_parfors_unsupported
    @unittest.skipIf(config.NUMBA_NUM_THREADS < 2, "Not enough CPU cores")
    def test_nested_parallelism_2(self):
        # check that get_thread_num is ok in nesting

        N = 5
        M = 17
        def get_impl(flag):

            if flag == True:
                dec = njit(parallel=True)
            elif flag == False:
                dec = njit(parallel=False)
            else:
                def dec(x): return x

            def gen(fid):
                @dec
                def child_func(buf):
                    M, N = buf.shape
                    set_num_threads(fid)  # set threads in child function
                    for i in prange(N):
                        buf[fid, i] = get_num_threads()
                return child_func

            child1 = gen(1)
            child2 = gen(2)
            child3 = gen(3)

            @dec
            def test_func(nthreads):
                acc = 0
                buf = np.zeros((M, N))
                set_num_threads(nthreads)
                for i in prange(M):
                    local_mask = 1 + i % mask
                    # when the threads exit the child functions they should have
                    # a TLS slot value of the local mask as it was set in
                    # child
                    if local_mask == 1:
                        child1(buf)
                        assert get_num_threads() == local_mask
                    elif local_mask == 2:
                        child2(buf)
                        assert get_num_threads() == local_mask
                    elif local_mask == 3:
                        child3(buf)
                        assert get_num_threads() == local_mask
                return buf
            return test_func

        mask = config.NUMBA_NUM_THREADS - 1
        set_num_threads(mask)
        pf_arr = get_impl(True)(mask)
        set_num_threads(mask)
        nj_arr = get_impl(False)(mask)
        set_num_threads(mask)
        py_arr = get_impl(None)(mask)

        np.testing.assert_equal(pf_arr, py_arr)
        np.testing.assert_equal(nj_arr, py_arr)

        # check the maths reconciles
        math_arr = np.zeros((M, N))
        for i in range(
                1, 4):  # there's branches on modulo mask but only 3 funcs
            math_arr[i, :] = i

        np.testing.assert_equal(math_arr, pf_arr)

    # this test can only run on OpenMP (providing OMP_MAX_ACTIVE_LEVELS is not
    # set or >= 2) and TBB backends
    @skip_parfors_unsupported
    @unittest.skipIf(config.NUMBA_NUM_THREADS < 2, "Not enough CPU cores")
    def test_nested_parallelism_3(self):
        if threading_layer() == 'workqueue':
            self.skipTest("workqueue is not threadsafe")

        # check that the right number of threads are present in nesting
        # this relies on there being a load of cores present
        BIG = 1000000

        @njit(parallel=True)
        def work(local_nt):
            tid = np.zeros(BIG)
            acc = 0
            set_num_threads(local_nt)
            for i in prange(BIG):
                acc += 1
                tid[i] = get_thread_num()
            return acc, np.unique(tid)

        @njit(parallel=True)
        def test_func(nthreads):
            acc = 0
            set_num_threads(nthreads)
            lens = np.zeros(nthreads)
            total = 0
            for i in prange(nthreads):
                my_acc, tids = work(nthreads + 1)
                lens[i] = len(tids)
                total += my_acc
            return total, np.unique(lens)

        NT = 2
        expected_acc = BIG * NT
        expected_thread_count = NT + 1

        got_acc, got_tc = test_func(NT)
        self.assertEqual(expected_acc, got_acc)
        np.testing.assert_equal(expected_thread_count, got_tc)

    def tearDown(self):
        set_num_threads(config.NUMBA_NUM_THREADS)


if __name__ == '__main__':
    unittest.main()
