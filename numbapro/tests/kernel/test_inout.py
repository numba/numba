from __future__ import print_function
import unittest
from contextlib import closing
import numpy as np
from numbapro.parallel.kernel import CU

def incr(tid, A):
    A[tid] += 1

class TestInOut(unittest.TestCase):
    def test_cpu(self):
#        print('cpu')
        with closing(CU('cpu')) as cu:
            for i in range(10):
                self._template(cu)

    def test_gpu(self):
#        print('gpu')
        with closing(CU('gpu')) as cu:
            for i in range(10):
                self._template(cu)

    def _template(self, cu):
        A = np.arange(10)    
        A0 = A.copy()
        dA = cu.inout(A)
        cu.enqueue(incr, ntid=dA.size, args=(dA,))
        cu.wait()
#        print(' A', A)
#        print('A0', A0)
        self.assertTrue(all(A == A0 + 1))


if __name__ == '__main__':
    unittest.main()

