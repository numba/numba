import unittest
import numpy as np
import math
from numba import *
from numbapro import cuda
import logging; logging.getLogger('numbapro').setLevel(1)


def cu_sqrt(A):
    i = cuda.grid(1)
    A[i] = math.sqrt(A[i])

def cu_exp(A):
    i = cuda.grid(1)
    A[i] = math.exp(A[i])

def cu_fabs(A):
    i = cuda.grid(1)
    A[i] = math.fabs(A[i])

def cu_log(A):
    i = cuda.grid(1)
    A[i] = math.log(A[i])

N = 10

class TestCudaMath(unittest.TestCase):
    def _template_f4(self, func, npfunc):
        cufunc = jit(argtypes=[f4[:]], target='gpu')(func)
        A = np.array(np.random.random(N), dtype=np.float32)
        Gold = npfunc(A)
        cufunc[(1,), A.shape](A)
        self.assertTrue(np.allclose(A, Gold))
    
    def _template_f8(self, func, npfunc):
        cufunc = jit(argtypes=[f8[:]], target='gpu')(func)
        A = np.array(np.random.random(N), dtype=np.float64)
        Gold = npfunc(A)
        cufunc[(1,), A.shape](A)
        self.assertTrue(np.allclose(A, Gold))

    def test_sqrt(self):
        self._template_f4(cu_sqrt, np.sqrt)
        self._template_f8(cu_sqrt, np.sqrt)

    def test_exp(self):
        self._template_f4(cu_exp, np.exp)
        self._template_f8(cu_exp, np.exp)

    def test_fabs(self):
        self._template_f4(cu_fabs, np.abs)
        self._template_f8(cu_fabs, np.abs)

    def test_log(self):
        self._template_f4(cu_log, np.log)
        self._template_f8(cu_log, np.log)

if __name__ == '__main__':
    unittest.main()


