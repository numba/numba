import unittest
from numba.cuda.testing import ContextResettingTestCase
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, xfail_with_cuda_python


@skip_on_cudasim('CUDA Profiler unsupported in the simulator')
@xfail_with_cuda_python
class TestProfiler(ContextResettingTestCase):
    def test_profiling(self):
        with cuda.profiling():
            a = cuda.device_array(10)
            del a

        with cuda.profiling():
            a = cuda.device_array(100)
            del a


if __name__ == '__main__':
    unittest.main()
