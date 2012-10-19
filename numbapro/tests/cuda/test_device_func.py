from numba import *
from numbapro import cuda
import numpy as np
import unittest

@cuda.jit(restype=f4, argtypes=[f4, f4], device=True, inline=True)
def cu_device_add(a, b):
    return a + b

@cuda.jit(argtypes=[f4[:], f4[:], f4[:]])
def cu_kernel_add(A, B, C):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    C[i] = cu_device_add(A[i], B[i])

class TestDeviceFunction(unittest.TestCase):
    def test_device_inlined(self):
        A = np.arange(10, dtype=np.float32)
        B = A.copy()
        C = np.empty_like(A)

        cu_kernel_add[(1,), (10,)](A, B, C)

        self.assertTrue((C == A + B).all())

if __name__ == '__main__':
    unittest.main()
