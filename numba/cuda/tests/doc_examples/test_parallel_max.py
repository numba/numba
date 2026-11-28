import unittest

from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout


@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestParallelMax(CUDATestCase):
    """
    Test parallel maximum computation
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

    def test_ex_parallel_max(self):
        # ex_parallel_max.import.begin
        import numpy as np
        from numba import cuda
        import math
        # ex_parallel_max.import.end

        # ex_parallel_max.kernel.begin
        @cuda.jit
        def max_reduce_kernel(input_array, output_array):
            """Find the maximum value in each block and store in output array."""
            # Shared memory to store partial results within a block
            temp = cuda.shared.array(shape=1024, dtype=np.float32)
            
            # Thread index within the block
            tx = cuda.threadIdx.x
            # Global thread index
            tid = cuda.grid(1)
            # Block size
            block_size = cuda.blockDim.x
            
            # Initialize shared memory
            if tid < len(input_array):
                temp[tx] = input_array[tid]
            else:
                temp[tx] = -np.inf  # Use negative infinity for non-data elements
            
            cuda.syncthreads()
            
            # Reduction in shared memory
            s = block_size // 2
            while s > 0:
                if tx < s and tid < len(input_array):
                    temp[tx] = max(temp[tx], temp[tx + s])
                cuda.syncthreads()
                s //= 2
            
            # Write the result for this block to global memory
            if tx == 0:
                output_array[cuda.blockIdx.x] = temp[0]
        # ex_parallel_max.kernel.end

        # ex_parallel_max.helper.begin
        def find_array_max(input_array):
            """Find the maximum value in an array using CUDA."""
            # Copy input data to device
            d_input = cuda.to_device(input_array)
            
            # Configure the blocks
            threads_per_block = 1024
            blocks_per_grid = math.ceil(len(input_array) / threads_per_block)
            
            # Create an output array for partial results
            d_output = cuda.device_array(blocks_per_grid, dtype=np.float32)
            
            # Launch the kernel
            max_reduce_kernel[blocks_per_grid, threads_per_block](d_input, d_output)
            
            # Copy the partial results back to host
            partial_results = d_output.copy_to_host()
            
            # Return the maximum value
            return np.max(partial_results)
        # ex_parallel_max.helper.end

        # ex_parallel_max.run.begin
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        
        # Create a test array
        N = 100000  # Smaller size for testing
        test_array = np.random.uniform(-1000, 1000, N).astype(np.float32)
        
        # Find the maximum using our CUDA function
        gpu_max = find_array_max(test_array)
        
        # Verify with numpy's max function
        cpu_max = np.max(test_array)
        
        print(f"GPU max: {gpu_max}")
        print(f"CPU max: {cpu_max}")
        print(f"Match: {np.isclose(gpu_max, cpu_max)}")
        # ex_parallel_max.run.end

        # Test that the GPU result matches the CPU result
        self.assertTrue(np.isclose(gpu_max, cpu_max))


if __name__ == "__main__":
    unittest.main()
