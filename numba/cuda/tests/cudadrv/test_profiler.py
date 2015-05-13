from __future__ import absolute_import, print_function
import numba.unittest_support as unittest
from numba.cuda.testing import CUDATestCase
from numba import cuda
from numba.cuda.testing import skip_on_cudasim


@skip_on_cudasim('CUDA Profiler unsupported in the simulator')
class TestProfiler(CUDATestCase):
    def test_profiling(self):
        with cuda.profiling():
            a = cuda.device_array(10)
            del a

        with cuda.profiling():
            a = cuda.device_array(100)
            del a


if __name__ == '__main__':
    unittest.main()

