from numba import *
from numbapro import cuda
import numpy as np
import unittest

@cuda.jit(argtypes=[f4[:], f4[:], f4[:]])
def cu_kernel_add(A, B, C):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    a = A[i]
    b = B[i]
    C[i] = a + b

class TestCudaAsync(unittest.TestCase):
    def test_cuda_async(self):
        A = np.arange(100, dtype=np.float32)
        B = A.copy()
        C = np.empty_like(A)

        with cuda.stream() as stream:
            orig_stream = stream

            dA = cuda.to_device(A, stream)
            dB = cuda.to_device(B, stream)
            dC = cuda.to_device(C, stream)

            stream = cu_kernel_add[(10,), (10,), stream](dA, dB, dC)

            self.assertTrue(orig_stream == stream)
            self.assertTrue(int(stream) != 0)

            dC.to_host()
        # synchronized here
        self.assertTrue((C == A + B).all())

if __name__ == '__main__':
    unittest.main()

