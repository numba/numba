from __future__ import absolute_import, print_function
from numba import ocl
import unittest


class TestCudaDetect(unittest.TestCase):
    def test_cuda_detect(self):
        # exercise the code path
        ocl.detect()


if __name__ == '__main__':
    unittest.main()
