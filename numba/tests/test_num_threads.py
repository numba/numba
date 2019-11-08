# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import numpy as np

from numba import njit, set_num_threads, get_num_threads, prange, config
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
        self.assertTrue(np.all(out == 2))

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
        self.assertTrue(np.all(out == mask))

    def tearDown(self):
        set_num_threads(config.NUMBA_NUM_THREADS)

if __name__ == '__main__':
    unittest.main()
