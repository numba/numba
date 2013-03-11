import unittest
from contextlib import closing
import numpy as np
from numbapro import CU, cuda

def mod(tid, A, b):
    A[tid] = A[tid] % b

class TestModulo(unittest.TestCase):
    def test_cpu(self):
        self._template('cpu')

    def test_gpu(self):
        if not cuda.is_available:
                print('Skipping CUDA test')
                return
        self._template('gpu')

    def _template(self, target):
        n = 10
        A = np.arange(n)
        A0 = A.copy()
        cu = CU(target)
        with closing(cu):
            dA = cu.inout(A)
            cu.enqueue(mod, ntid=A.size, args=(A, n))
            cu.wait()
        assert np.allclose(A, A0 % n)

if __name__ == '__main__':
    unittest.main()
