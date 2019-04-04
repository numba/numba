# (Some) Tests for numbers.py
from __future__ import print_function, absolute_import, division

from numba import njit
from numba.errors import TypingError
from .support import TestCase

import numpy as np


@njit
def view32(scalar_val, initial_type, astype):
    if initial_type == "uint32":
        return np.uint32(scalar_val).view(astype)
    elif initial_type == "int32":
        return np.int32(scalar_val).view(astype)
    elif initial_type == "float32":
        return np.float32(scalar_val).view(astype)


@njit
def view64(scalar_val, initial_type, astype):
    if initial_type == "uint64":
        return np.uint64(scalar_val).view(astype)
    elif initial_type == "int64":
        return np.int64(scalar_val).view(astype)
    elif initial_type == "float64":
        return np.float64(scalar_val).view(astype)


class TestViewIntFloat(TestCase):
    def test_32_bits(self):
        dtypes = (np.uint32, np.int32, np.float32)
        inputs = ((1,   "uint32",    (1, 1, 1.401298464324817e-45)),
                  (-1,  "int32",     (4294967295, -1, np.nan)),
                  (1.0, "float32",   (1065353216, 1065353216, 1.0)))
        for value, initial_type, expected in inputs:
            for target_type, result in zip(dtypes, expected):
                if not np.isnan(result):
                    self.assertEqual(view32(value, initial_type, target_type),
                                     target_type(result))
                else:
                    self.assertTrue(np.isnan(view32(value,
                                                    initial_type,
                                                    target_type)))

    def test_64_bits(self):
        dtypes = (np.uint64, np.int64, np.float64)
        inputs = ((1,   "uint64",    (1, 1, 5e-324)),
                  (-1,  "int64",     (18446744073709551615, -1, np.nan)),
                  (1.0, "float64",   (4607182418800017408,
                                      4607182418800017408,
                                      1.0))
                  )
        for value, initial_type, expected in inputs:
            for target_type, result in zip(dtypes, expected):
                if not np.isnan(result):
                    self.assertEqual(view64(value, initial_type, target_type),
                                     target_type(result))
                else:
                    self.assertTrue(np.isnan(view64(value,
                                                    initial_type,
                                                    target_type)))

    def test_exceptions(self):
        with self.assertRaises(TypingError) as e:
            view32(1, "int32", np.int64)
        self.assertIn("Changing the dtype of a 0d array is only supported "
                      "if the itemsize is unchanged",
                      str(e.exception))
        with self.assertRaises(TypingError) as e:
            view64(1, "int64", np.int32)
        self.assertIn("Changing the dtype of a 0d array is only supported "
                      "if the itemsize is unchanged",
                      str(e.exception))
