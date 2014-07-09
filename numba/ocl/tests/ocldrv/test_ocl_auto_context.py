from __future__ import print_function, absolute_import
import numpy as np
from numba import ocl
import numba.unittest_support as unittest


class TestCudaAutoContext(unittest.TestCase):
    def test_auto_context(self):
        """
        This caused problems in the CUDA driver at some point.
        Make sure that there is no regression in OpenCL
        """
        A = np.arange(10, dtype=np.float32)
        newA = np.empty_like(A)
        dA = ocl.to_device(A)

        dA.copy_to_host(newA)
        self.assertTrue(np.allclose(A, newA))

if __name__ == '__main__':
    unittest.main()
