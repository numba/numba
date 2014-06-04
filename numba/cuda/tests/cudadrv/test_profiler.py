from __future__ import absolute_import, print_function
import numba.unittest_support as unittest
from numba.cuda.testing import CUDATestCase
from numba import cuda


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

