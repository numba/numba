import unittest

from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout


@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestConvolution(CUDATestCase):
    """
    Test image convolution using CUDA
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

    def test_ex_convolution(self):
        # ex_convolution.import.begin
        import numpy as np
        from numba import cuda
        import math
        # ex_convolution.import.end

        # ex_convolution.kernel.begin
        @cuda.jit
        def convolution_kernel(image, kernel, output):
            """
            Apply a 2D convolution filter to an image using CUDA.
            
            Parameters:
            -----------
            image : 2D array
                Input image
            kernel : 2D array
                Convolution kernel
            output : 2D array
                Output image
            """
            # Get thread indices
            i, j = cuda.grid(2)
            
            # Get kernel dimensions
            k_rows, k_cols = kernel.shape
            k_center_y, k_center_x = k_rows // 2, k_cols // 2
            
            # Only process valid image pixels
            if i < output.shape[0] and j < output.shape[1]:
                val = 0.0
                # Apply the convolution
                for k_i in range(k_rows):
                    for k_j in range(k_cols):
                        # Get the corresponding image pixel
                        i_pos = i + (k_i - k_center_y)
                        j_pos = j + (k_j - k_center_x)
                        
                        # Check if the position is valid (zero padding)
                        if (0 <= i_pos < image.shape[0] and 
                            0 <= j_pos < image.shape[1]):
                            val += image[i_pos, j_pos] * kernel[k_i, k_j]
                
                # Store the result
                output[i, j] = val
        # ex_convolution.kernel.end

        # ex_convolution.helper.begin
        def apply_convolution(image, kernel):
            """
            Apply a convolution filter to an image using CUDA.
            
            Parameters:
            -----------
            image : 2D numpy array
                Input image
            kernel : 2D numpy array
                Convolution kernel
            
            Returns:
            --------
            2D numpy array
                Filtered image
            """
            # Ensure the inputs are float32
            image = image.astype(np.float32)
            kernel = kernel.astype(np.float32)
            
            # Allocate memory on the device
            d_image = cuda.to_device(image)
            d_kernel = cuda.to_device(kernel)
            d_output = cuda.device_array_like(image)
            
            # Configure the grid
            threads_per_block = (16, 16)
            blocks_per_grid_x = math.ceil(image.shape[0] / threads_per_block[0])
            blocks_per_grid_y = math.ceil(image.shape[1] / threads_per_block[1])
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            
            # Launch the kernel
            convolution_kernel[blocks_per_grid, threads_per_block](
                d_image, d_kernel, d_output
            )
            
            # Copy the result back to host
            result = d_output.copy_to_host()
            
            return result
        # ex_convolution.helper.end

        # ex_convolution.run.begin
        # Create a small sample image for testing (a simple gradient)
        image_size = 64
        image = np.zeros((image_size, image_size), dtype=np.float32)
        for i in range(image_size):
            for j in range(image_size):
                image[i, j] = (i + j) / (2 * image_size)
        
        # Define a simple convolution kernel (Gaussian blur)
        gaussian = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32) / 16.0
        
        # Apply the convolution
        blurred = apply_convolution(image, gaussian)
        
        print(f"Original image shape: {image.shape}")
        print(f"Filtered image shape: {blurred.shape}")
        print(f"Original min: {image.min()}, max: {image.max()}")
        print(f"Blurred min: {blurred.min()}, max: {blurred.max()}")
        # ex_convolution.run.end

        # Test that the shapes match
        self.assertEqual(image.shape, blurred.shape)
        
        # Test the convolution by comparing against NumPy's implementation
        from scipy import signal
        cpu_blurred = signal.convolve2d(image, gaussian, mode='same')
        np.testing.assert_allclose(blurred, cpu_blurred, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
