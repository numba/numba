from __future__ import print_function

import numpy as np

from numba import cuda
from numba.cuda.testing import unittest


CONST1D = np.arange(10, dtype=np.float64) / 2.
CONST2D = np.asfortranarray(
                np.arange(100, dtype=np.int32).reshape(10, 10))
CONST3D = ((np.arange(5*5*5, dtype=np.complex64).reshape(5, 5, 5) + 1j) /
            2j)


def cuconst(A):
    C = cuda.const.array_like(CONST1D)
    i = cuda.grid(1)
    A[i] = C[i]


def cuconst2d(A):
    C = cuda.const.array_like(CONST2D)
    i, j = cuda.grid(2)
    A[i, j] = C[i, j]


def cuconst3d(A):
    C = cuda.const.array_like(CONST3D)
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    k = cuda.threadIdx.z
    A[i, j, k] = C[i, j, k]


class TestCudaConstantMemory(unittest.TestCase):
    def test_const_array(self):
        jcuconst = cuda.jit('void(float64[:])')(cuconst)
        self.assertTrue('.const' in jcuconst.ptx)
        A = np.empty_like(CONST1D)
        jcuconst[2, 5](A)
        self.assertTrue(np.all(A == CONST1D))

    def test_const_array_2d(self):
        jcuconst2d = cuda.jit('void(int32[:,:])')(cuconst2d)
        self.assertTrue('.const' in jcuconst2d.ptx)
        A = np.empty_like(CONST2D, order='C')
        jcuconst2d[(2,2), (5,5)](A)
        self.assertTrue(np.all(A == CONST2D))

    def test_const_array_3d(self):
        jcuconst3d = cuda.jit('void(complex64[:,:,:])')(cuconst3d)
        self.assertTrue('.const' in jcuconst3d.ptx)
        A = np.empty_like(CONST3D, order='F')
        jcuconst3d[1, (5, 5, 5)](A)
        self.assertTrue(np.all(A == CONST3D))


if __name__ == '__main__':
    unittest.main()
