from numba import cuda, int32

from numba.cuda.testing import unittest


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


if __name__ == '__main__':
    unittest.main()
