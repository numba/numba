from __future__ import absolute_import, print_function
from numba import cuda
from numba.cuda.testing import unittest


class TestCudaDetect(unittest.TestCase):
    def test_cuda_detect(self):
        # exercise the code path
        cuda.detect()


if __name__ == '__main__':
    unittest.main()
