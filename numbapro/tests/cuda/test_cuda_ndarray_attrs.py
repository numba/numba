import unittest
import numpy as np
from numba import *
from numbapro import cuda
import support

@jit(argtypes=[int32[:]], target='gpu')
def cu_ndarray_shape(A):
    A[0] = A.shape[0]

@jit(argtypes=[int32[:]], target='gpu')
def cu_ndarray_strides(A):
    A[0] = A.strides[0]

class TestNDArrayAttrs(support.CudaTestCase):
    def test_ndarray_shape(self):
        A = np.zeros(1, dtype=np.int32)
        cu_ndarray_shape[(1,), A.shape](A)
        self.assertTrue(A[0] == A.shape[0])

    def test_ndarray_strides(self):
        A = np.zeros(1, dtype=np.int32)
        cu_ndarray_strides[(1,), A.shape](A)
        self.assertTrue(A[0] == A.strides[0])


if __name__ == '__main__':
    unittest.main()


