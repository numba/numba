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


class TestCudaSMem(unittest.TestCase):
    def test_array_float(self):
        A = np.array(np.random.random(256), dtype=np.float32)
        A = np.arange(256, dtype=np.float32)
        Gold = A * 2
        
        cu_array_double[(1,), (A.shape[0],)](A)
        
        for i, (got, expect) in enumerate(zip(A, Gold)):
            self.assertEqual(got, expect, "%s != %s at i=%d" % (got, expect, i))

if __name__ == '__main__':
    unittest.main()


