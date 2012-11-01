import unittest
import numpy as np
import math
from numba import *
from numbapro import cuda
#import logging; logging.getLogger().setLevel(1)


def cu_sqrt(A):
    i = cuda.grid(1)
    A[i] = math.sqrt(A[i])

def cu_exp(A):
    i = cuda.grid(1)
    A[i] = math.exp(A[i])

class TestCudaMath(unittest.TestCase):
    def _template(self, func, npfunc):
        cufunc = jit(argtypes=[f4[:]], target='gpu')(func)
        A = np.array(np.random.random(32), dtype=np.float32)
        Gold = npfunc(A)
        cufunc[(1,), A.shape](A)
        np.allclose(A, Gold)

    def test_sqrt(self):
        self._template(cu_sqrt, np.sqrt)

    def test_exp(self):
        self._template(cu_exp, np.exp)

if __name__ == '__main__':
    unittest.main()


