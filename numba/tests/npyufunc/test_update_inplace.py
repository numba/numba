# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import unittest

import numpy as np
from numba import guvectorize
from numba.tests.support import TestCase


def py_replace_2nd(x_t, y_1):
    for t in range(0, x_t.shape[0], 2):
        x_t[t] = y_1[0]


class TestUpdateInplace(TestCase):

    def _run_test_for_gufunc(self, gufunc, assume_f4_works=True):
        # f8 (works)
        x_t = np.zeros(10, 'f8')
        gufunc(x_t, 2)
        np.testing.assert_equal(x_t[1::2], 0)
        np.testing.assert_equal(x_t[::2], 2)

        # f4 (works)
        x_t = np.zeros(10, 'f4')
        gufunc(x_t, 2)
        if assume_f4_works:
            np.testing.assert_equal(x_t[1::2], 0)
            np.testing.assert_equal(x_t[::2], 2)
        else:
            np.testing.assert_equal(x_t, 0)

    def test_update_inplace(self):
        # test without writable_args
        gufunc = guvectorize(['void(f8[:], f8[:])'], '(t),()',
                             nopython=True)(py_replace_2nd)
        self._run_test_for_gufunc(gufunc, assume_f4_works=False)

        # test with writable_args
        gufunc = guvectorize(['void(f8[:], f8[:])'], '(t),()',
                             nopython=True, writable_args=(0,))(py_replace_2nd)
        self._run_test_for_gufunc(gufunc)

    def test_update_inplace_with_cache(self):
        # test with writable_args
        gufunc = guvectorize(['void(f8[:], f8[:])'], '(t),()',
                             nopython=True, writable_args=(0,),
                             cache=True)(py_replace_2nd)
        # 2nd time it is loaded from cache
        gufunc = guvectorize(['void(f8[:], f8[:])'], '(t),()',
                             nopython=True, writable_args=(0,),
                             cache=True)(py_replace_2nd)
        self._run_test_for_gufunc(gufunc)

    def test_update_inplace_parallel(self):
        # test with writable_args
        gufunc = guvectorize(['void(f8[:], f8[:])'], '(t),()',
                             nopython=True, writable_args=(0,),
                             target='parallel')(py_replace_2nd)
        self._run_test_for_gufunc(gufunc)


if __name__ == '__main__':
    unittest.main()
