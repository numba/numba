import unittest
import numpy as np
from numba import *
from numbapro import cuda
import support

@jit(argtypes=[f4[:]], target='gpu')
def cu_array_double(dst):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x
    i = tid + blkid * blkdim
    dst[i] *= 2

@jit(argtypes=[i4[:], i4], target='gpu')
def cu_array_scalar_assign(dst, scalar):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x
    i = tid + blkid * blkdim
    dst[i] += scalar


class TestCudaJitArray(support.CudaTestCase):
    def test_array_double(self):
        A = np.array(np.random.random(256), dtype=np.float32)
        Gold = A * 2

        cu_array_double[(A.shape[0],), (1,)](A)

        for i, (got, expect) in enumerate(zip(A, Gold)):
            self.assertEqual(got, expect, "%s != %s at i=%d" % (got, expect, i))

    def test_array_scalar_assign(self):
        A = np.array(np.random.random(256), dtype=np.int32)
        scalar = 0xdead
        Gold = A + scalar

        cu_array_scalar_assign[(A.shape[0],), (1,)](A, scalar)

        for i, (got, expect) in enumerate(zip(A, Gold)):
            self.assertEqual(got, expect, "%s != %s at i=%d" % (got, expect, i))


if __name__ == '__main__':
    unittest.main()


