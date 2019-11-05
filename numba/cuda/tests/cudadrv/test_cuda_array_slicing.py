from __future__ import print_function

from itertools import product

import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, SerialMixin


class CudaArrayIndexing(SerialMixin, unittest.TestCase):
    def test_index_1d(self):
        arr = np.arange(10)
        darr = cuda.to_device(arr)
        for i in range(arr.size):
            self.assertEqual(arr[i], darr[i])

    def test_index_2d(self):
        arr = np.arange(3 * 4).reshape(3, 4)
        darr = cuda.to_device(arr)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                self.assertEqual(arr[i, j], darr[i, j])

    def test_index_3d(self):
        arr = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        darr = cuda.to_device(arr)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    self.assertEqual(arr[i, j, k], darr[i, j, k])


class CudaArrayStridedSlice(SerialMixin, unittest.TestCase):

    def test_strided_index_1d(self):
        arr = np.arange(10)
        darr = cuda.to_device(arr)
        for i in range(arr.size):
            np.testing.assert_equal(arr[i::2], darr[i::2].copy_to_host())

    def test_strided_index_2d(self):
        arr = np.arange(6 * 7).reshape(6, 7)
        darr = cuda.to_device(arr)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                np.testing.assert_equal(arr[i::2, j::2],
                                        darr[i::2, j::2].copy_to_host())

    def test_strided_index_3d(self):
        arr = np.arange(6 * 7 * 8).reshape(6, 7, 8)
        darr = cuda.to_device(arr)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    np.testing.assert_equal(
                        arr[i::2, j::2, k::2],
                        darr[i::2, j::2, k::2].copy_to_host())


class CudaArraySlicing(SerialMixin, unittest.TestCase):
    def test_prefix_1d(self):
        arr = np.arange(5)
        darr = cuda.to_device(arr)
        for i in range(arr.size):
            expect = arr[i:]
            got = darr[i:].copy_to_host()
            self.assertTrue(np.all(expect == got))

    def test_prefix_2d(self):
        arr = np.arange(3 ** 2).reshape(3, 3)
        darr = cuda.to_device(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                expect = arr[i:, j:]
                sliced = darr[i:, j:]
                self.assertEqual(expect.shape, sliced.shape)
                self.assertEqual(expect.strides, sliced.strides)
                got = sliced.copy_to_host()
                self.assertTrue(np.all(expect == got))

    def test_select_3d_first_two_dim(self):
        arr = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        darr = cuda.to_device(arr)
        # Select first dimension
        for i in range(arr.shape[0]):
            expect = arr[i]
            sliced = darr[i]
            self.assertEqual(expect.shape, sliced.shape)
            self.assertEqual(expect.strides, sliced.strides)
            got = sliced.copy_to_host()
            self.assertTrue(np.all(expect == got))
        # Select second dimension
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                expect = arr[i, j]
                sliced = darr[i, j]
                self.assertEqual(expect.shape, sliced.shape)
                self.assertEqual(expect.strides, sliced.strides)
                got = sliced.copy_to_host()
                self.assertTrue(np.all(expect == got))

    def test_select_f(self):
        a = np.arange(5 * 6 * 7).reshape(5, 6, 7, order='F')
        da = cuda.to_device(a)

        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                self.assertTrue(np.array_equal(da[i, j, :].copy_to_host(),
                                               a[i, j, :]))
            for j in range(a.shape[2]):
                self.assertTrue(np.array_equal(da[i, :, j].copy_to_host(),
                                               a[i, :, j]))
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                self.assertTrue(np.array_equal(da[:, i, j].copy_to_host(),
                                               a[:, i, j]))

    def test_select_c(self):
        a = np.arange(5 * 6 * 7).reshape(5, 6, 7, order='C')
        da = cuda.to_device(a)

        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                self.assertTrue(np.array_equal(da[i, j, :].copy_to_host(),
                                               a[i, j, :]))
            for j in range(a.shape[2]):
                self.assertTrue(np.array_equal(da[i, :, j].copy_to_host(),
                                               a[i, :, j]))
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                self.assertTrue(np.array_equal(da[:, i, j].copy_to_host(),
                                               a[:, i, j]))

    def test_prefix_select(self):
        arr = np.arange(5 * 7).reshape(5, 7, order='F')

        darr = cuda.to_device(arr)
        self.assertTrue(np.all(darr[:1, 1].copy_to_host() == arr[:1, 1]))

    def test_negative_slicing_1d(self):
        arr = np.arange(10)
        darr = cuda.to_device(arr)
        for i, j in product(range(-10, 10), repeat=2):
            np.testing.assert_array_equal(arr[i:j],
                                          darr[i:j].copy_to_host())

    def test_negative_slicing_2d(self):
        arr = np.arange(12).reshape(3, 4)
        darr = cuda.to_device(arr)
        for x, y, w, s in product(range(-4, 4), repeat=4):
            np.testing.assert_array_equal(arr[x:y, w:s],
                                          darr[x:y, w:s].copy_to_host())

    def test_empty_slice_1d(self):
        arr = np.arange(5)
        darr = cuda.to_device(arr)
        for i in range(darr.shape[0]):
            np.testing.assert_array_equal(darr[i:i].copy_to_host(), arr[i:i])
        # empty slice of empty slice
        self.assertFalse(darr[:0][:0].copy_to_host())
        # out-of-bound slice just produces empty slices
        np.testing.assert_array_equal(darr[:0][:1].copy_to_host(),
                                      arr[:0][:1])
        np.testing.assert_array_equal(darr[:0][-1:].copy_to_host(),
                                      arr[:0][-1:])

    def test_empty_slice_2d(self):
        arr = np.arange(5 * 7).reshape(5, 7)
        darr = cuda.to_device(arr)
        np.testing.assert_array_equal(darr[:0].copy_to_host(), arr[:0])
        np.testing.assert_array_equal(darr[3, :0].copy_to_host(), arr[3, :0])
        # empty slice of empty slice
        self.assertFalse(darr[:0][:0].copy_to_host())
        # out-of-bound slice just produces empty slices
        np.testing.assert_array_equal(darr[:0][:1].copy_to_host(), arr[:0][:1])
        np.testing.assert_array_equal(darr[:0][-1:].copy_to_host(),
                                      arr[:0][-1:])


class CudaArraySetting(SerialMixin, unittest.TestCase):
    """
    Most of the slicing logic is tested in the cases above, so these
    tests focus on the setting logic.
    """

    def test_scalar(self):
        arr = np.arange(5 * 7).reshape(5, 7)
        darr = cuda.to_device(arr)
        arr[2, 2] = 500
        darr[2, 2] = 500
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_rank(self):
        arr = np.arange(5 * 7).reshape(5, 7)
        darr = cuda.to_device(arr)
        arr[2] = 500
        darr[2] = 500
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_broadcast(self):
        arr = np.arange(5 * 7).reshape(5, 7)
        darr = cuda.to_device(arr)
        arr[:, 2] = 500
        darr[:, 2] = 500
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_array_assign_column(self):
        arr = np.arange(5 * 7).reshape(5, 7)
        darr = cuda.to_device(arr)
        _400 = np.full(shape=7, fill_value=400)
        arr[2] = _400
        darr[2] = _400
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_array_assign_row(self):
        arr = np.arange(5 * 7).reshape(5, 7)
        darr = cuda.to_device(arr)
        _400 = np.full(shape=5, fill_value=400)
        arr[:, 2] = _400
        darr[:, 2] = _400
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_array_assign_subarray(self):
        arr = np.arange(5 * 6 * 7).reshape(5, 6, 7)
        darr = cuda.to_device(arr)
        _400 = np.full(shape=(6, 7), fill_value=400)
        arr[2] = _400
        darr[2] = _400
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_array_assign_deep_subarray(self):
        arr = np.arange(5 * 6 * 7 * 8).reshape(5, 6, 7, 8)
        darr = cuda.to_device(arr)
        _400 = np.full(shape=(5, 6, 8), fill_value=400)
        arr[:, :, 2] = _400
        darr[:, :, 2] = _400
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_array_assign_all(self):
        arr = np.arange(5 * 7).reshape(5, 7)
        darr = cuda.to_device(arr)
        _400 = np.full(shape=(5, 7), fill_value=400)
        arr[:] = _400
        darr[:] = _400
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_strides(self):
        arr = np.ones(20)
        darr = cuda.to_device(arr)
        arr[::2] = 500
        darr[::2] = 500
        np.testing.assert_array_equal(darr.copy_to_host(), arr)

    def test_incompatible_highdim(self):
        darr = cuda.to_device(np.arange(5 * 7))

        with self.assertRaises(ValueError) as e:
            darr[:] = np.ones(shape=(1, 2, 3))

        self.assertIn(
            member=str(e.exception),
            container=[
                "Can't assign 3-D array to 1-D self",  # device
                "could not broadcast input array from shape (2,3) "
                "into shape (35)",  # simulator
            ])

    def test_incompatible_shape(self):
        darr = cuda.to_device(np.arange(5))

        with self.assertRaises(ValueError) as e:
            darr[:] = [1, 3]

        self.assertIn(
            member=str(e.exception),
            container=[
                "Can't copy sequence with size 2 to array axis 0 with "
                "dimension 5",  # device
                "cannot copy sequence with size 2 to array axis with "
                "dimension 5",  # simulator
            ])


if __name__ == '__main__':
    unittest.main()
