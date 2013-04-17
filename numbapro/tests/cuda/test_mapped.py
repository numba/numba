from numbapro import cuda
from numba import f4, void
import support
from timeit import default_timer as timer
import numpy as np
import unittest

@cuda.jit(void(f4[:]))
def sum(A):
    i = cuda.grid(1)
    A[i] *= 2

class TestMapped(support.CudaTestCase):

    def test_mapped(self):
        A = np.arange(4*1024*1024, dtype=np.float32)
        print 'Bytes: ', A.nbytes / (2 ** 20), 'MB'
        A0 = A.copy()
        blockdim = 1024, 1, 1
        griddim = A.size // 1024, 1
        stream = cuda.stream()
        with cuda.mapped(A) as ptr:
            # Array A is mapped to GPU directly
            sum[griddim, blockdim, stream](ptr)

        stream.synchronize()
        
        self.assertTrue(np.allclose(A0 * 2, A))


if __name__ == '__main__':
    unittest.main()

