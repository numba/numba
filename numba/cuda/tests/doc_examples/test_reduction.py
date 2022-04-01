import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout


@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestReduction(CUDATestCase):
    """
    Test shared memory reduction
    """

    def setUp(self):
        # Prevent output from this test showing up when running the test suite
        self._captured_stdout = captured_stdout()
        self._captured_stdout.__enter__()
        super().setUp()

    def tearDown(self):
        # No exception type, value, or traceback
        self._captured_stdout.__exit__(None, None, None)
        super().tearDown()

    def test_ex_reduction(self):
        # ex_reduction.import.begin
        from numba import cuda
        import numpy as np
        from numba.types import int32

        # ex_reduction.import.end
        
        # ex_reduction.allocate.begin
        # generate data
        a = cuda.to_device(np.arange(1024))
        nelem = len(a)
        # ex_reduction.allocate.end

        # ex_reduction.kernel.begin
        @cuda.jit
        def array_sum(data, size):
            tid = cuda.threadIdx.x
            if tid < size:
                i = cuda.grid(1)
                
                # declare an array in shared memory
                shr = cuda.shared.array(nelem, int32)
                shr[tid] = data[i]
                
                # make sure every thread has written its value to shared memory
                # before we start reducing
                cuda.syncthreads()
                
                s = 1
                while s < cuda.blockDim.x:
                    if tid % (2 * s) == 0:
                        # stride by `s` and add
                        shr[tid] += shr[tid + s]
                    s *= 2
                    cuda.syncthreads()
                    
                # after the loop, the zeroth element contains the sum
                if tid == 0:
                    data[tid] = shr[tid]
        # ex_reduction.kernel.end

        # ex_reduction.launch.begin
        array_sum.forall(len(a))(a, len(a))
        print(a[0]) # array(523776)
        sum(np.arange(1024)) # 523776
        # ex_reduction.launch.end

        

if __name__ == '__main__':
    unittest.main()
