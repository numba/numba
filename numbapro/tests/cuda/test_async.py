from numba import *
from numbapro import cuda
import numpy as np
import unittest
import support

@cuda.jit(argtypes=[f4[:], f4[:], f4[:]])
def cu_kernel_add(A, B, C):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    a = A[i]
    b = B[i]
    C[i] = a + b

class TestCudaAsync(support.CudaTestCase):
    def test_cuda_async(self):
        A = np.arange(100, dtype=np.float32)
        B = A.copy()
        C = np.empty_like(A)

        stream = cuda.stream()
        with stream.auto_synchronize():
            dA = cuda.to_device(A, stream)
            dB = cuda.to_device(B, stream)
            dC = cuda.to_device(C, stream)

            cu_kernel_add[(10,), (10,), stream](dA, dB, dC)

            self.assertFalse((C == A + B).all())

            dC.to_host(stream)

        # synchronized here
        self.assertTrue((C == A + B).all())

if __name__ == '__main__':
    unittest.main()

