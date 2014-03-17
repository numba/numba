from __future__ import print_function, absolute_import
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest


class TestCudaAutoContext(unittest.TestCase):
    def test_auto_context(self):
        """A problem was revealed by a customer that the use cuda.to_device
        does not create a CUDA context.
        This tests the problem
        """
        A = np.arange(10, dtype=np.float32)
        newA = np.empty_like(A)
        dA = cuda.to_device(A)

        dA.copy_to_host(newA)
        self.assertTrue(np.allclose(A, newA))

if __name__ == '__main__':
    unittest.main()
