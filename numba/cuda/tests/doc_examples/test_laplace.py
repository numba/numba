import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout


@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestLaplace(CUDATestCase):
    """
    Test simple vector addition
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

    def test_ex_laplace(self):
        # ex_laplace.import.begin
        from numba import cuda
        import numpy as np
        # ex_laplace.import.end
        
        # ex_laplace.allocate.begin
        size = 100
        data = np.zeros(size)
        # make one piece in the very middle very hot
        data[int(size/2)] = 10000
        data_gpu = cuda.to_device(data)
        # ex_laplace.allocate.end

        # ex_laplace.kernel.begin
        @cuda.jit
        def solve_heat_equation(data, size, timesteps, k):
            i = cuda.grid(1)
            for step in range(timesteps):
                # get the current temperature associated with this segment
                curr_temp = data[i]
                
                # apply formula from finite difference equation
                if i == 0:
                    # Left wall is held at T = 0
                    next_temp = curr_temp + k * (data[i + 1] - (2 * curr_temp))
                elif i == size - 1:
                    # Right wall is held at T = 0
                    next_temp = curr_temp + k * (data[i - 1] - (2 * curr_temp))
                else:
                    next_temp = curr_temp + k * (data[i - 1] - (2 * curr_temp) + data[i + 1])
                data[i] = next_temp
                
                # wait for every thread to write before moving on
                cuda.syncthreads()
        # ex_laplace.kernel.end

        # ex_laplace.launch.begin
        solve_heat_equation.forall(len(data))(data_gpu, len(data_gpu), 10000, 0.25)
        # ex_laplace.launch.end

        # ex_laplace.print.begin
        print(data_gpu.copy_to_host())
        # ex_laplace.print.end

if __name__ == '__main__':
    unittest.main()
