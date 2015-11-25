from __future__ import print_function
import numpy as np
from numba import cuda
from numba import unittest_support as unittest


class TestRewriteArray(unittest.TestCase):
    def test_rewrite_numpy_empty(self):
        size = 10

        @cuda.jit
        def foo(out, out_strides):
            arr = np.empty(size, dtype=np.int32)
            for i in range(size):
                arr[i] = cuda.threadIdx.x + i

            for i in range(size):
                out[cuda.threadIdx.x, i] = arr[i]

            if cuda.threadIdx.x == 0:
                for i, s in enumerate(arr.strides):
                    out_strides[i] = s

        threads = 4
        out = np.zeros((threads, size), dtype=np.int32)
        out_strides = np.zeros(1, dtype=np.intp)

        foo[1, threads](out, out_strides)

        self.assertEqual(out_strides[0], out.itemsize)

        np.testing.assert_equal(out[0], np.arange(size, dtype=np.int32))

        for i in range(1, out.shape[0]):
            np.testing.assert_equal(out[i], 1 + out[i - 1])

    def test_rewrite_numpy_empty_nd(self):

        @cuda.jit
        def foo(out, out_strides):
            arr = np.empty((2, 3, 6), dtype=np.int32)
            c = 0
            for x in range(arr.shape[0]):
                for y in range(arr.shape[1]):
                    for z in range(arr.shape[2]):
                        arr[x, y, z] = cuda.threadIdx.x + c
                        c += 1

            for x in range(arr.shape[0]):
                for y in range(arr.shape[1]):
                    for z in range(arr.shape[2]):
                        out[cuda.threadIdx.x, x, y, z] = arr[x, y, z]

            if cuda.threadIdx.x == 0:
                for i, s in enumerate(arr.strides):
                    out_strides[i] = s

        threads = 4
        out = np.zeros((threads, 2, 3, 6), dtype=np.int32)
        out_strides = np.zeros(3, dtype=np.intp)

        foo[1, threads](out, out_strides)

        self.assertEqual(out.strides[1:], tuple(out_strides))

        np.testing.assert_equal(out[0].ravel(), np.arange(out[0].size,
                                                          dtype=np.int32))

        for i in range(1, out.shape[0]):
            np.testing.assert_equal(out[i], 1 + out[i - 1])


if __name__ == '__main__':
    unittest.main()
