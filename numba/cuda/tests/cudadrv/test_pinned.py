import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase


REPEAT = 25


class TestPinned(ContextResettingTestCase):

    def _run_copies(self, A):
        A0 = np.copy(A)

        stream = cuda.stream()
        ptr = cuda.to_device(A, copy=False, stream=stream)
        ptr.copy_to_device(A, stream=stream)
        ptr.copy_to_host(A, stream=stream)
        stream.synchronize()

        self.assertTrue(np.allclose(A, A0))

    def test_pinned(self):
        A = np.arange(2*1024*1024) # 16 MB
        with cuda.pinned(A):
            self._run_copies(A)

    def test_unpinned(self):
        A = np.arange(2*1024*1024) # 16 MB
        self._run_copies(A)


if __name__ == '__main__':
    unittest.main()

