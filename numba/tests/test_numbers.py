# (Some) Tests for numbers.py
from __future__ import print_function, absolute_import, division

from numba import njit
from numba.errors import TypingError
from .support import TestCase

import numpy as np


def gen_view(a,b):
    def impl(x):
        return a(x).view(b)
    return impl


class TestViewIntFloat(TestCase):

    def do_testing(self, inputs, dtypes):
        for value, initial_type, expected in inputs:
            for target_type, result in zip(dtypes, expected):
                view = njit(gen_view(initial_type, target_type))
                if not np.isnan(result):
                    self.assertEqual(view(value), target_type(result))
                else:
                    self.assertTrue(np.isnan(view(value)))

    def test_32_bits(self):
        dtypes = (np.uint32, np.int32, np.float32)
        #        Value  Initial Type   Expected answers using dtypes
        inputs = ((1,   np.uint32,    (1, 1, 1.401298464324817e-45)),
                  (-1,  np.int32,     (4294967295, -1, np.nan)),
                  (1.0, np.float32,   (1065353216, 1065353216, 1.0)))
        self.do_testing(inputs, dtypes)

    def test_64_bits(self):
        dtypes = (np.uint64, np.int64, np.float64)
        #        Value  Initial Type   Expected answers using dtypes
        inputs = ((1,   np.uint64,    (1, 1, 5e-324)),
                  (-1,  np.int64,     (18446744073709551615, -1, np.nan)),
                  (1.0, np.float64,   (4607182418800017408,
                                       4607182418800017408,
                                       1.0))
                  )
        self.do_testing(inputs, dtypes)

    def test_exceptions(self):
        with self.assertRaises(TypingError) as e:
            view = njit(gen_view(np.int32, np.int64))
            view(1)
        self.assertIn("Changing the dtype of a 0d array is only supported "
                      "if the itemsize is unchanged",
                      str(e.exception))
        with self.assertRaises(TypingError) as e:
            view = njit(gen_view(np.int64, np.int32))
            view(1)
        self.assertIn("Changing the dtype of a 0d array is only supported "
                      "if the itemsize is unchanged",
                      str(e.exception))
