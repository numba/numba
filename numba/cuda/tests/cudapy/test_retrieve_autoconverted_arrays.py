from __future__ import print_function, absolute_import, division

import numpy as np

from numba import unittest_support as unittest
from numba import cuda, float32
from numba.cuda.testing import SerialMixin


def set_array_to_three(arr):
    arr[0] = 3

def set_record_to_three(rec):
    rec[0]['b'] = 3


recordtype = np.dtype(
    [('b', np.int32)],
    align=True
)


class TestRetrieveAutoconvertedArrays(SerialMixin, unittest.TestCase):
    def test_array_true(self):
        host_arr = np.zeros(1, dtype=np.int64)
        cuda.jit(retrieve_autoconverted_arrays=True)(set_array_to_three)(host_arr)
        self.assertEqual(3, host_arr[0])

    def test_array_false(self):
        host_arr = np.zeros(1, dtype=np.int64)
        cuda.jit(retrieve_autoconverted_arrays=False)(set_array_to_three)(host_arr)
        self.assertEqual(0, host_arr[0])

    def test_array_default(self):
        host_arr = np.zeros(1, dtype=np.int64)
        cuda.jit()(set_array_to_three)(host_arr)
        self.assertEqual(3, host_arr[0])

    def test_record_default_true(self):
        host_rec = np.zeros(1, dtype=recordtype)
        cuda.jit(retrieve_autoconverted_arrays=True)(set_record_to_three)(host_rec)
        self.assertEqual(3, host_rec[0]['b'])

    def test_record_default_false(self):
        host_rec = np.zeros(1, dtype=recordtype)
        cuda.jit(retrieve_autoconverted_arrays=False)(set_record_to_three)(host_rec)
        self.assertEqual(0, host_rec[0]['b'])

    def test_record_default(self):
        host_rec = np.zeros(1, dtype=recordtype)
        cuda.jit()(set_record_to_three)(host_rec)
        self.assertEqual(3, host_rec[0]['b'])



if __name__ == '__main__':
    unittest.main()
