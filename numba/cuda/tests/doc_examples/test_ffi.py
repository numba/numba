# Contents in this file are referenced from the sphinx-generated docs.
# "magictoken" is used for markers as beginning and ending of example text.

import unittest
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
                                skip_unless_cuda_python)


@skip_unless_cuda_python('NVIDIA Binding needed for NVRTC')
@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestFFI(CUDATestCase):
    def test_ex_linking_cu(self):
        # magictoken.ex_linking_cu.begin
        from numba import cuda
        import numpy as np
        import os

        # Declaration of the foreign function
        mul = cuda.declare_device('mul_f32_f32', 'float32(float32, float32)')

        # Path to the source containing the foreign function
        # (here assumed to be in a subdirectory called "ffi")
        basedir = os.path.dirname(os.path.abspath(__file__))
        functions_cu = os.path.join(basedir, 'ffi', 'functions.cu')

        # Kernel that links in functions.cu and calls mul
        @cuda.jit(link=[functions_cu])
        def multiply_vectors(r, x, y):
            i = cuda.grid(1)

            if i < len(r):
                r[i] = mul(x[i], y[i])

        # Generate random data
        N = 32
        np.random.seed(1)
        x = np.random.rand(N).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        r = np.zeros_like(x)

        # Run the kernel
        multiply_vectors[1, 32](r, x, y)

        # Sanity check - ensure the results match those expected
        np.testing.assert_array_equal(r, x * y)
        # magictoken.ex_linking_cu.end


if __name__ == '__main__':
    unittest.main()
