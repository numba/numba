# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import unittest

import numpy as np
from numba import guvectorize
from numba.tests.support import TestCase


class TestUpdateInplace(TestCase):

    def test_update_inplace(self):
        def replace_2nd(x_t, y_1):
            for t in range(0, x_t.shape[0], 2):
                x_t[t] = y_1[0]

        # test without writable_args
        gufunc = guvectorize(['void(f8[:], f8[:])'], '(t),()',
                             nopython=True)(replace_2nd)
        # f8 (works)
        x_t = np.zeros(10, 'f8')
        gufunc(x_t, 2)
        np.testing.assert_equal(x_t[1::2], 0)
        np.testing.assert_equal(x_t[::2], 2)

        # f4 (does not work)
        x_t = np.zeros(10, 'f4')
        gufunc(x_t, 2)
        np.testing.assert_equal(x_t, 0)

        # test with writable_args
        gufunc = guvectorize(['void(f8[:], f8[:])'], '(t),()',
                             nopython=True, writable_args=(0,))(replace_2nd)
        # f8 (works)
        x_t = np.zeros(10, 'f8')
        gufunc(x_t, 2)
        np.testing.assert_equal(x_t[1::2], 0)
        np.testing.assert_equal(x_t[::2], 2)

        # f4 (works)
        x_t = np.zeros(10, 'f4')
        gufunc(x_t, 2)
        np.testing.assert_equal(x_t[1::2], 0)
        np.testing.assert_equal(x_t[::2], 2)


if __name__ == '__main__':
    unittest.main()
