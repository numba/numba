from numbapro import cuda
import support
from timeit import default_timer as timer
import numpy as np
import unittest

@cuda.autojit
def fill(A):
    i = cuda.grid(1)
    A[i] = i

class TestDeviceArray(support.CudaTestCase):

    def test_device_array_like(self):
        n = 10
        A = np.arange(n)
        dA = cuda.device_array_like(A)
        self.assertTrue(A.shape == dA.shape)

        fill[(1,), (n,)](dA)
        
        B = np.empty_like(A)
        dA.copy_to_host(B)
        self.assertTrue(np.all(np.arange(n) == B))
#
    def test_devary_array_strides(self):
        shape = 4, 5, 6
        # test C order
        exp = np.empty(shape, order='C').strides
        got = cuda.device_array(shape, order='C').strides
        self.assertEqual(exp, got)

        # test F order
        exp = np.empty(shape, order='F').strides
        got = cuda.device_array(shape, order='F').strides
        self.assertEqual(exp, got)

if __name__ == '__main__':
    unittest.main()

