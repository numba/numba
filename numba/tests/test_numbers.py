# (Some) Tests for numbers.py
from __future__ import print_function, absolute_import, division

from numba import njit
from numba.errors import TypingError
from .support import TestCase

import numpy as np


@njit
def view(scalar_val, astype):
    return scalar_val.view(astype)


class TestViewIntFloat(TestCase):

    def test_32_bits(self):
        dtypes = (np.uint32, np.int32, np.float32)
        inputs = ((np.uint32(1),    (1, 1, 1.401298464324817e-45)),
                  (np.int32(-1),    (4294967295, -1, np.nan)),
                  (np.float32(1.0), (1065353216, 1065353216, 1.0)))
        for input_, expected in inputs:
            for type_, result in zip(dtypes, expected):
                if not np.isnan(result):
                    self.assertEqual(view(input_, type_), type_(result))
                else:
                    self.assertTrue(np.isnan(view(input_, type_)))

    def test_64_bits(self):
        dtypes = (np.uint64, np.int64, np.float64)
        inputs = ((np.uint64(1),    (1, 1, 5e-324)),
                  (np.int64(-1),    (18446744073709551615, -1, np.nan)),
                  (np.float64(1.0), (4607182418800017408,
                                     4607182418800017408,
                                     1.0))
                  )
        for input_, expected in inputs:
            for type_, result in zip(dtypes, expected):
                if not np.isnan(result):
                    self.assertEqual(view(input_, type_), type_(result))
                else:
                    self.assertTrue(np.isnan(view(input_, type_)))

    def test_exceptions(self):
        with self.assertRaises(TypingError) as e:
            view(np.int32(1), np.int64)
        self.assertIn("Changing the dtype of a 0d array is only supported "
                      "if the itemsize is unchanged",
                      str(e.exception))
        with self.assertRaises(TypingError) as e:
            view(np.int64(1), np.int32)
        self.assertIn("Changing the dtype of a 0d array is only supported "
                      "if the itemsize is unchanged",
                      str(e.exception))
