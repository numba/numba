from __future__ import print_function
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest


class CudaArrayIndexing(unittest.TestCase):
    def test_index_1d(self):
        arr = np.arange(10)
        darr = cuda.to_device(arr)
        for i in range(arr.size):
            self.assertEqual(arr[i], darr[i])

    def test_index_2d(self):
        arr = np.arange(9).reshape(3, 3)
        darr = cuda.to_device(arr)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                self.assertEqual(arr[i, j], darr[i, j])

    def test_index_3d(self):
        arr = np.arange(3 ** 3).reshape(3, 3, 3)
        darr = cuda.to_device(arr)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    self.assertEqual(arr[i, j, k], darr[i, j, k])


class CudaArraySlicing(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
