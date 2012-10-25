import unittest
import numpy as np
from numba import *
from numbapro import cuda


@jit(argtypes=[f4[:], f4[:]], target='gpu')
def cu_array_double(dst):
    smem = cuda.shared.array(shape=(32,), dtype=f4)

    '''
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x
    i = tid + blkid * blkdim
    smem[i] = dst[i]          # store in smem
    dst[i] = smem[i] * 2      # use smem'''
    

class TestCudaSMem(unittest.TestCase):
    def test_array_float(self):
        A = np.array(np.random.random(256), dtype=np.float32)
        Gold = A * 2
        
        smem = cuda.smem(shape=A.shape, dtype=np.float32)
        cu_array_double[(A.shape[0],), (1,)](A, smem)
        
        for i, (got, expect) in enumerate(zip(A, Gold)):
            self.assertEqual(got, expect, "%s != %s at i=%d" % (got, expect, i))

if __name__ == '__main__':
    unittest.main()


