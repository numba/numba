from numba import cuda, int32

from numba.cuda.testing import unittest

import numpy as np

class TestSharedMemoryIssue(unittest.TestCase):
    def test_issue_953_sm_linkage_conflict(self):
        @cuda.jit(device=True)
        def inner():
            inner_arr = cuda.shared.array(1, dtype=int32)

        @cuda.jit
        def outer():
            outer_arr = cuda.shared.array(1, dtype=int32)
            inner()

        outer()

    def _check_shared_array_size(self, shape, expected):
        @cuda.jit
        def s(a):
            arr = cuda.shared.array(shape, dtype=int32)
            a[0] = arr.size

        result = np.zeros(1, dtype=np.int32)
        s(result)
        self.assertEqual(result[0], expected)

    def test_issue_1051_shared_size_broken_1d(self):
        self._check_shared_array_size(2, 2)

    def test_issue_1051_shared_size_broken_2d(self):
        self._check_shared_array_size((2, 3), 6)

    def test_issue_1051_shared_size_broken_3d(self):
        self._check_shared_array_size((2, 3, 4), 24)


if __name__ == '__main__':
    unittest.main()
