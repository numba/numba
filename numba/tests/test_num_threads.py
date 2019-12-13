# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import numpy as np

from numba import (njit, set_num_threads, get_num_threads, prange, config,
                   threading_layer, guvectorize)
from numba.npyufunc.parallel import _get_thread_id
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

        self.assertEqual(get_n(), max_threads)
        set_num_threads(2)
        self.assertEqual(get_n(), 2)
        set_num_threads(max_threads)
        self.assertEqual(get_n(), max_threads)

        @njit
        def set_get_n(n):
            set_num_threads(n)
            return get_num_threads()

        self.assertEqual(set_get_n(2), 2)
        self.assertEqual(set_get_n(max_threads), max_threads)

        with self.assertRaises(ValueError):
            set_get_n(0)

        with self.assertRaises(ValueError):
            set_get_n(max_threads + 1)

    @skip_parfors_unsupported
    @unittest.skipIf(config.NUMBA_NUM_THREADS < 2, "Not enough CPU cores")
    def test_set_num_threads_basic_guvectorize(self):
        max_threads = config.NUMBA_NUM_THREADS

        @guvectorize(['void(int64[:])'],
                     '(n)',
                     nopython=True,
                     target='parallel')
        def get_n(x):
            x[:] = get_num_threads()

        x = np.zeros((5000000,), dtype=np.int64)
        get_n(x)
        np.testing.assert_equal(x, max_threads)
        set_num_threads(2)
        x = np.zeros((5000000,), dtype=np.int64)
        get_n(x)
        np.testing.assert_equal(x, 2)
        set_num_threads(max_threads)
        x = np.zeros((5000000,), dtype=np.int64)
        get_n(x)
        np.testing.assert_equal(x, max_threads)

        @guvectorize(['void(int64[:])'],
                     '(n)',
                     nopython=True,
                     target='parallel')
        def set_get_n(n):
            set_num_threads(n[0])
            n[:] = get_num_threads()

        x = np.zeros((5000000,), dtype=np.int64)
        x[0] = 2
        set_get_n(x)
        np.testing.assert_equal(x, 2)
        x = np.zeros((5000000,), dtype=np.int64)
        x[0] = max_threads
        set_get_n(x)
        np.testing.assert_equal(x, max_threads)

        with self.assertRaises(ValueError):
            set_get_n(np.array([0]))

        with self.assertRaises(ValueError):
            set_get_n(np.array([max_threads + 1]))

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

        @guvectorize(['void(int64[:])'],
                     '(n)',
                     nopython=True,
                     target='parallel')
        def test_gufunc(x):
            x[:] = get_num_threads()


        out = test_func()
        np.testing.assert_equal(out, 2)

        x = np.zeros((5000000,), dtype=np.int64)
        test_gufunc(x)
        np.testing.assert_equal(x, 2)

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
    def test_set_num_threads_inside_guvectorize(self):
        # Test set_num_threads inside a jitted guvectorize function
        @guvectorize(['void(int64[:])'],
                     '(n)',
                     nopython=True,
                     target='parallel')
        def test_func(x):
            set_num_threads(x[0])
            x[:] = get_num_threads()

        x = np.zeros((5000000,), dtype=np.int64)
        mask = 2
        x[0] = mask
        test_func(x)
        np.testing.assert_equal(x, mask)

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
                    buf[i] = _get_thread_id()
                return len(np.unique(buf)), get_num_threads()

            out = test_func()
            self.assertEqual(out, (mask, mask))

            @guvectorize(['void(int64[:], int64[:])'],
                         '(n), (m)',
                         nopython=True,
                         target='parallel')
            def test_gufunc(x, out):
                x[:] = _get_thread_id()
                out[0] = get_num_threads()

            # Reshape to force parallelism
            x = np.full((5000000,), -1, dtype=np.int64).reshape((100, 50000))
            out = np.zeros((1,), dtype=np.int64)
            test_gufunc(x, out)
            np.testing.assert_equal(out, np.array([mask]))
            self.assertEqual(len(np.unique(x)), mask)

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
                    buf[i] = _get_thread_id()
                return len(np.unique(buf)), get_num_threads()

            out = test_func()
            self.assertEqual(out, (mask, mask))


            @guvectorize(['void(int64[:], int64[:])'],
                         '(n), (m)',
                         nopython=True,
                         target='parallel')
            def test_gufunc(x, out):
                set_num_threads(mask)
                x[:] = _get_thread_id()
                out[0] = get_num_threads()

            # Reshape to force parallelism
            x = np.full((5000000,), -1, dtype=np.int64).reshape((100, 50000))
            out = np.zeros((1,), dtype=np.int64)
            test_gufunc(x, out)
            np.testing.assert_equal(out, np.array([mask]))
            self.assertEqual(len(np.unique(x)), mask)

    # this test can only run on OpenMP (providing OMP_MAX_ACTIVE_LEVELS is not
    # set or >= 2) and TBB backends
    @skip_parfors_unsupported
    @unittest.skipIf(config.NUMBA_NUM_THREADS < 2, "Not enough CPU cores")
    def test_nested_parallelism_1(self):
        if threading_layer() == 'workqueue':
            self.skipTest("workqueue is not threadsafe")

        # check that get_num_threads is ok in nesting
        mask = config.NUMBA_NUM_THREADS - 1

        N = config.NUMBA_NUM_THREADS
        M = 2*config.NUMBA_NUM_THREADS

        @njit(parallel=True)
        def child_func(buf, fid):
            M, N = buf.shape
            for i in prange(N):
                buf[fid, i] = get_num_threads()

        def get_test(test_type):
            if test_type == 'njit':
                def test_func(nthreads, py_func=False):
                    @njit(parallel=True)
                    def _test_func(nthreads):
                        acc = 0
                        buf = np.zeros((M, N))
                        set_num_threads(nthreads)
                        for i in prange(M):
                            local_mask = 1 + i % mask
                            set_num_threads(local_mask)  # set threads in parent function
                            if local_mask < N:
                                child_func(buf, local_mask)
                            acc += get_num_threads()
                        return acc, buf
                    if py_func:
                        return _test_func.py_func(nthreads)
                    else:
                        return _test_func(nthreads)

            elif test_type == 'guvectorize':
                def test_func(nthreads, py_func=False):
                    def _test_func(acc, buf, local_mask):
                        set_num_threads(nthreads)
                        set_num_threads(local_mask[0])  # set threads in parent function
                        if local_mask[0] < N:
                            child_func(buf, local_mask[0])
                        acc[0] += get_num_threads()

                    buf = np.zeros((M, N), dtype=np.int64)
                    acc = np.array([0])
                    local_mask = (1 + np.arange(M) % mask).reshape((M, 1))
                    if not py_func:
                        _test_func = guvectorize(['void(int64[:], int64[:, :], int64[:])'],
                                                 '(k), (n, m), (p)', nopython=True,
                                                 target='parallel')(_test_func)
                    else:
                        _test_func = guvectorize(['void(int64[:], int64[:, :], int64[:])'],
                                                 '(k), (n, m), (p)', forceobj=True)(_test_func)
                    _test_func(acc, buf, local_mask)
                    return acc, buf

            return test_func

        for test_type in ['njit', 'guvectorize']:
            test_func = get_test(test_type)

            got_acc, got_arr = test_func(mask)
            exp_acc, exp_arr = test_func(mask, py_func=True)
            self.assertEqual(exp_acc, got_acc, test_type)
            np.testing.assert_equal(exp_arr, got_arr)

            # check the maths reconciles
            math_acc = np.sum(1 + np.arange(M) % mask)
            self.assertEqual(math_acc, got_acc)
            math_arr = np.zeros((M, N))
            for i in range(1, N):  # there's branches on 1, ..., num_threads - 1
                math_arr[i, :] = i
            np.testing.assert_equal(math_arr, got_arr)

    # this test can only run on OpenMP (providing OMP_MAX_ACTIVE_LEVELS is not
    # set or >= 2) and TBB backends
    @skip_parfors_unsupported
    @unittest.skipIf(config.NUMBA_NUM_THREADS < 2, "Not enough CPU cores")
    def test_nested_parallelism_2(self):
        if threading_layer() == 'workqueue':
            self.skipTest("workqueue is not threadsafe")

        # check that get_num_threads is ok in nesting

        N = config.NUMBA_NUM_THREADS + 1
        M = 4*config.NUMBA_NUM_THREADS + 1

        def get_impl(flag):

            if flag == True:
                dec = njit(parallel=True)
            elif flag == False:
                dec = njit(parallel=False)
            else:
                def dec(x): return x

            @dec
            def child(buf, fid):
                M, N = buf.shape
                set_num_threads(fid)  # set threads in child function
                for i in prange(N):
                    buf[fid, i] = get_num_threads()

            @dec
            def test_func(nthreads):
                buf = np.zeros((M, N))
                set_num_threads(nthreads)
                for i in prange(M):
                    local_mask = 1 + i % mask
                    # when the threads exit the child functions they should have
                    # a TLS slot value of the local mask as it was set in
                    # child
                    if local_mask < config.NUMBA_NUM_THREADS:
                        child(buf, local_mask)
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
        for i in range(1, config.NUMBA_NUM_THREADS):  # there's branches on modulo mask but only NUMBA_NUM_THREADS funcs
            math_arr[i, :] = i

        np.testing.assert_equal(math_arr, pf_arr)

    # this test can only run on OpenMP (providing OMP_MAX_ACTIVE_LEVELS is not
    # set or >= 2) and TBB backends
    @skip_parfors_unsupported
    @unittest.skipIf(config.NUMBA_NUM_THREADS < 3, "Not enough CPU cores")
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
                tid[i] = _get_thread_id()
            return acc, np.unique(tid)

        @njit(parallel=True)
        def test_func(nthreads):
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
