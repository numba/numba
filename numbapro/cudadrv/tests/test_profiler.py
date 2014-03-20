from __future__ import absolute_import, print_function
from numba.cuda.testing import CUDATestCase
from numba import cuda
import unittest

class TestProfiler(CUDATestCase):
    def test_profiling(self):
        with cuda._profiling():
            a = cuda.device_array(10)
            del a

        with cuda._profiling():
            a = cuda.device_array(100)
            del a


if __name__ == '__main__':
    unittest.main()

