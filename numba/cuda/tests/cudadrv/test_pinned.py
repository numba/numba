from __future__ import print_function, division, absolute_import
from timeit import default_timer as timer
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase


REPEAT = 25


class TestPinned(CUDATestCase):

    def _template(self, name, A):
        A0 = np.copy(A)

        s = timer()
        stream = cuda.stream()
        ptr = cuda.to_device(A, copy=False, stream=stream)

        ptr.copy_to_device(A, stream=stream)

        ptr.copy_to_host(A, stream=stream)
        stream.synchronize()

        e = timer()

        self.assertTrue(np.allclose(A, A0))

        elapsed = e - s
        return elapsed

    def test_pinned(self):
        A = np.arange(2*1024*1024) # 16 MB
        total = 0
        with cuda.pinned(A):
            for i in range(REPEAT):
                total += self._template('pinned', A)

    def test_unpinned(self):
        A = np.arange(2*1024*1024) # 16 MB
        total = 0
        for i in range(REPEAT):
            total += self._template('unpinned', A)


if __name__ == '__main__':
    unittest.main()

