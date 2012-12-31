from numbapro import cuda
import support
from timeit import default_timer as timer
import numpy as np
import unittest


class TestPinned(support.CudaTestCase):

    def _template(self, name, A):
        A0 = np.copy(A)

        s = timer()
        stream = cuda.stream()
        ptr = cuda.to_device(A, copy=False, stream=stream)

        ptr.to_device(stream=stream)

        ptr.to_host(stream=stream)
        stream.synchronize()

        e = timer()

        self.assertTrue(np.allclose(A, A0))
        
        print name, e - s

    def test_pinned(self):
        A = np.arange(2*1024*1024) # 16 MB
        with cuda.pagelock(A):
            self._template('pinned', A)

    def test_unpinned(self):
        A = np.arange(2*1024*1024) # 16 MB
        self._template('unpinned', A)


if __name__ == '__main__':
    unittest.main()

