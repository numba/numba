from __future__ import print_function
import unittest
from contextlib import closing
import numpy as np
from numbapro import CU, cuda

# do saxpy in two kernels

def product(tid, A, B, Prod):
    Prod[tid] = A[tid] * B[tid]

def sum(tid, A, B, Sum):
    Sum[tid] = A[tid] + B[tid]

class TestSaxpy(unittest.TestCase):
    def test_cpu(self):
        with closing(CU('cpu')) as cu:
            self._template(cu)

    def test_gpu(self):
        if not cuda.is_available:
            print('Skipping CUDA test')
            return

        with closing(CU('gpu')) as cu:
            self._template(cu)
    
    def _template(self, cu):
        n = 100
        A = np.arange(n)
        B = np.arange(n)
        C = np.arange(n)
#        print(A)
#        print(B)
#        print(C)

        D = np.empty(n)
        
        dA = cu.input(A)
        dB = cu.input(B)
        dC = cu.input(C)
        dProd = cu.scratch_like(D)
        dSum  = cu.output(D)

        cu.enqueue(product, ntid=dProd.size, args=(dA, dB, dProd))
        cu.enqueue(sum, 	ntid=dSum.size,  args=(dProd, dC, dSum))

        cu.wait()
#        print(D.size, D)

        # check
        self.assertTrue(np.allclose(A * B + C, D))


if __name__ == '__main__':
    unittest.main()
