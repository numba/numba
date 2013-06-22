from numbapro import cuda
from numba import *
import numpy as np
import unittest

class TestCudaAutoContext(unittest.TestCase):
    def test_auto_context(self):
        '''A problem was revealed by a customer that the use cuda.to_device
        does not create a CUDA context.
        This tests the problem
        '''
        A = np.arange(10, dtype=np.float32)
        dA = cuda.to_device(A)

        @cuda.jit(void(f4[:]))
        def foo(A):
            i = cuda.grid(1)
            A[i] = A[i] * 2

        orig = A.copy()
        foo[(1,), A.shape](dA)
        dA.copy_to_host(A)

        self.assertTrue(np.allclose(A, orig * 2), (A, orig * 2))

if __name__ == '__main__':
    unittest.main()