import unittest
from contextlib import closing
import numpy as np
from numbapro.parallel.kernel import CU

def zero(tid, ary):
    ary[tid] = 0

def binning(tid, inA, inB, outC, outD):
    vala = inA[tid]
    valb = inB[tid]
    if vala > valb:
        outC[tid] = vala
    else:
        outD[tid] = valb

class TestBinning(unittest.TestCase):
    def test_cpu(self):
        with closing(CU('cpu')) as cu:
            self._template(cu)

    def test_gpu(self):
        with closing(CU('gpu')) as cu:
            self._template(cu)

    def _template(self, cu):
        n = 10000
        np.random.seed(0xcafe)
        A = np.random.random(n)
        B = np.random.random(n)
#        print(A)
#        print(B)

        C = np.empty(n)
        D = np.empty(n)

        dA = cu.input(A)
        dB = cu.input(B)
        dC = cu.output(C)
        dD = cu.output(D)

        cu.enqueue(zero, ntid=dC.size, args=(dC,))
        cu.enqueue(zero, ntid=dD.size, args=(dD,))
        cu.enqueue(binning, ntid=dC.size, args=(dA, dB, dC, dD))
        
        cu.wait()

        # check
        goldC = np.zeros(n)
        goldD = np.zeros(n)
        for i in xrange(dC.size):
            binning(i, A, B, goldC, goldD)

        self.assertTrue(np.all(C == goldC))
        self.assertTrue(np.all(D == goldD))


if __name__ == '__main__':
    unittest.main()
