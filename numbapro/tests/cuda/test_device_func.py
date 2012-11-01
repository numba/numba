from numba import *
from numbapro import cuda
import numpy as np
import unittest
import math

@cuda.jit(restype=f4, argtypes=[f4, f4], device=True, inline=True)
def cu_device_add(a, b):
    return a + b

@cuda.jit(argtypes=[f4[:], f4[:], f4[:]])
def cu_kernel_add(A, B, C):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    C[i] = cu_device_add(A[i], B[i])

@cuda.jit(argtypes=[f4[:], f4[:], f4[:]])
def cu_kernel_add_2(A, B, C):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    C[i] = cu_device_add(A[i], B[i])
    C[i] = cu_device_add(A[i], C[i])

@cuda.jit(restype=f4, argtypes=[f4], device=True, inline=True)
def cu_device_exp(x):
    return math.exp(x)

@cuda.jit(argtypes=[f4[:]])
def cu_kernel_exp(A):
    i = cuda.grid(1)
    A[i] = cu_device_exp(A[i])

class TestDeviceFunction(unittest.TestCase):
    def test_device_inlined(self):
        A = np.arange(10, dtype=np.float32)
        B = A.copy()
        C = np.empty_like(A)

        cu_kernel_add[(1,), (10,)](A, B, C)

        self.assertTrue((C == A + B).all())

    def test_device_inlined_2(self):
        A = np.arange(10, dtype=np.float32)
        B = A.copy()
        C = np.empty_like(A)

        cu_kernel_add_2[(1,), (10,)](A, B, C)

        self.assertTrue((C == A + A + B).all())

    def test_device_math(self):
        A = np.arange(32, dtype=np.float32)
        Gold = np.exp(A)
        cu_kernel_exp[(1,), A.shape](A)
        self.assertTrue(np.allclose(A, Gold))

if __name__ == '__main__':
    unittest.main()
