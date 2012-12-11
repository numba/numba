from numbapro import cuda
import support
from timeit import default_timer as timer
import numpy as np
import unittest

class TestPinned(support.CudaTestCase):
    def test_pinned(self):
        A = np.arange(2*1024*1024) # 16 MB
        A0 = A.copy()
        s = timer()
        for i in range(10):
            dA = cuda.to_device(A, pinned=False)
            dA.to_host()
            self.assertTrue(np.allclose(A, A0))
        e = timer()
        print "pinned", e - s

    def test_unpinned(self):
        A = np.arange(2*1024*1024) # 16 MB
        A0 = A.copy()
        s = timer()
        for i in range(10):
            dA = cuda.to_device(A, pinned=False)
            dA.to_host()
            self.assertTrue(np.allclose(A, A0))
        e = timer()
        print "unpinned", e - s

if __name__ == '__main__':
    unittest.main()

