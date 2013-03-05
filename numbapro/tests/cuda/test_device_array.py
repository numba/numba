from numbapro import cuda
import support
from timeit import default_timer as timer
import numpy as np
import unittest

@cuda.autojit
def fill(A):
    i = cuda.grid(1)
    A[i] = i

class TestDeviceOnly(support.CudaTestCase):

    def test_device_array(self):
        n = 10
        A = np.arange(n)
        dA = cuda.device_array_like(A)
        self.assertTrue(A.shape == dA.shape)

        fill[(1,), (n,)](dA)
        
        B = np.empty_like(A)
        dA.copy_to_host(B)
        self.assertTrue(np.all(np.arange(n) == B))

if __name__ == '__main__':
    unittest.main()

