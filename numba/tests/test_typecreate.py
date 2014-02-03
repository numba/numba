from __future__ import print_function, absolute_import, division
from numba import unittest_support as unittest
from numba import types


class TestTypeCreate(unittest.TestCase):
    def test_array_c(self):
        self.assertEqual(types.int32[::1].layout, 'C')
        self.assertEqual(types.int32[:, ::1].layout, 'C')
        self.assertEqual(types.int32[:, :, ::1].layout, 'C')

        self.assertTrue(types.int32[::1].is_c_contig)
        self.assertTrue(types.int32[:, ::1].is_c_contig)
        self.assertTrue(types.int32[:, :, ::1].is_c_contig)

    def test_array_f(self):
        self.assertEqual(types.int32[::1, :].layout, 'F')
        self.assertEqual(types.int32[::1, :, :].layout, 'F')

        self.assertTrue(types.int32[::1].is_f_contig)
        self.assertTrue(types.int32[::1, :].is_f_contig)
        self.assertTrue(types.int32[::1, :, :].is_f_contig)

    def test_array_a(self):
        self.assertEqual(types.int32[:].layout, 'A')
        self.assertEqual(types.int32[:, :].layout, 'A')
        self.assertEqual(types.int32[:, :, :].layout, 'A')

        self.assertFalse(types.int32[:].is_contig)
        self.assertFalse(types.int32[:, :].is_contig)
        self.assertFalse(types.int32[:, :, :].is_contig)


if __name__ == '__main__':
    unittest.main()
