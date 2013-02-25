import unittest
import numpy as np
import math
from numba import *
from numbapro import cuda
import support
#import logging; logging.getLogger('numbapro').setLevel(1)

def cu_abs(A):
    i = cuda.grid(1)
    A[i] = abs(A[i])

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

def cu_pow(A, B):
    i = cuda.grid(1)
    A[i] = A[i] ** B[i]

N = 10

class TestCudaMath(support.CudaTestCase):
    def _template_f4(self, func, npfunc):
        cufunc = jit(argtypes=[f4[:]], target='gpu')(func)
        A = np.array(np.random.random(N), dtype=np.float32)
        Gold = npfunc(A)
        cufunc[(1,), A.shape](A)
        self.assertTrue(np.allclose(A, Gold))

    def _template_f4f4(self, func, npfunc):
        cufunc = jit(argtypes=[f4[:], f4[:]], target='gpu')(func)
        A = np.array(np.random.random(N), dtype=np.float32)
        B = np.array(np.random.random(N), dtype=np.float32)
        Gold = npfunc(A, B)
        cufunc[(1,), A.shape](A, B)
        self.assertTrue(np.allclose(A, Gold))

    def _template_f8(self, func, npfunc):
        cufunc = jit(argtypes=[f8[:]], target='gpu')(func)
        A = np.array(np.random.random(N), dtype=np.float64)
        Gold = npfunc(A)
        cufunc[(1,), A.shape](A)
        self.assertTrue(np.allclose(A, Gold))

    def _template_f8f8(self, func, npfunc):
        cufunc = jit(argtypes=[f8[:], f8[:]], target='gpu')(func)
        A = np.array(np.random.random(N), dtype=np.float64)
        B = np.array(np.random.random(N), dtype=np.float64)
        Gold = npfunc(A, B)
        cufunc[(1,), A.shape](A, B)
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

    def test_pow(self):
        self._template_f4f4(cu_pow, np.power)
        self._template_f8f8(cu_pow, np.power)

    def test_abs(self):
        self._template_f4f4(cu_abs, np.abs)

if __name__ == '__main__':
    unittest.main()


