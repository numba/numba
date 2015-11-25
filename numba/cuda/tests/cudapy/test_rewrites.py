from __future__ import print_function
import numpy as np
from numba import cuda
from numba import unittest_support as unittest


class TestRewriteArray(unittest.TestCase):
    def test_rewrite_numpy_empty(self):
        size = 10

        @cuda.jit
        def foo(out):
            arr = np.empty(size, dtype=np.int32)
            for i in range(size):
                arr[i] = cuda.threadIdx.x + i

            for i in range(size):
                out[cuda.threadIdx.x, i] = arr[i]

        threads = 4
        out = np.zeros((threads, size), dtype=np.int32)

        foo[1, threads](out)

        np.testing.assert_equal(out[0], np.arange(size, dtype=np.int32))

        for i in range(1, out.shape[0]):
            np.testing.assert_equal(out[i], 1 + out[i - 1])


if __name__ == '__main__':
    unittest.main()
