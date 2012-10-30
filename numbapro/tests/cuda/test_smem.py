import unittest
import numpy as np
from numba import *
from numbapro import cuda


@jit(argtypes=[f4[:]], target='gpu')
def cu_array_double(dst):
    smem = cuda.shared.array(shape=(256,), dtype=f4)
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x
    i = tid + blkid * blkdim
    smem[i] = dst[i]          # store in smem
    dst[i] = smem[i] * 2      # use smem


@jit(argtypes=[f4[:, :]], target='gpu')
def cu_array_double_2d(dst):
    smem = cuda.shared.array(shape=(32, 16), dtype=f4)
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    smem[j, i] = dst[j, i]          # store in smem
    dst[j, i] = smem[j, i] * 2      # use smem


class TestCudaSMem(unittest.TestCase):
    def test_array_double(self):
        A = np.array(np.random.random(256), dtype=np.float32)
        Gold = A * 2
        
        cu_array_double[(1,), (A.shape[0],)](A)
        
        for i, (got, expect) in enumerate(zip(A, Gold)):
            self.assertEqual(got, expect, "%s != %s at i=%d" % (got, expect, i))
    
    def test_array_double_2d(self):
        shape = 16, 32
        A = np.array(np.random.random(shape), dtype=np.float32)
        Gold = A * 2
    
        cu_array_double_2d[(1,), tuple(reversed(shape))](A)
        
        for i, (got, expect) in enumerate(zip(A.flatten(), Gold.flatten())):
            self.assertEqual(got, expect, "%s != %s at i=%d" % (got, expect, i))

if __name__ == '__main__':
    unittest.main()


