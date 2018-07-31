from __future__ import print_function

import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, SerialMixin

CONST1D = np.arange(10, dtype=np.float64) / 2.
CONST2D = np.asfortranarray(
    np.arange(100, dtype=np.int32).reshape(10, 10))
CONST3D = ((np.arange(5 * 5 * 5, dtype=np.complex64).reshape(5, 5, 5) + 1j) /
           2j)
CONST_RECORD = np.array(
    [(1.0, 2), (3.0, 4)],
    dtype=[('x', float), ('y', int)])
CONST_RECORD_ALIGN = np.array(
    [(1, 2, 3, 0xDEADBEEF, 8), (4, 5, 6, 0xBEEFDEAD, 10)],
    dtype=np.dtype(
        dtype=[
            ('a', np.uint8),
            ('b', np.uint8),
            ('x', np.uint8),
            ('y', np.uint32),
            ('z', np.uint8),
        ],
        align=True))


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


def cuconstRec(A, B):
    C = cuda.const.array_like(CONST_RECORD)
    i = cuda.grid(1)
    A[i] = C[i]['x']
    B[i] = C[i]['y']


def cuconstRecAlign(A, B, C, D, E):
    Z = cuda.const.array_like(CONST_RECORD_ALIGN)
    i = cuda.grid(1)
    A[i] = Z[i]['a']
    B[i] = Z[i]['b']
    C[i] = Z[i]['x']
    D[i] = Z[i]['y']
    E[i] = Z[i]['z']


class TestCudaConstantMemory(SerialMixin, unittest.TestCase):
    def test_const_array(self):
        jcuconst = cuda.jit('void(float64[:])')(cuconst)
        self.assertTrue('.const' in jcuconst.ptx)
        A = np.zeros_like(CONST1D)
        jcuconst[2, 5](A)
        self.assertTrue(np.all(A == CONST1D))

    def test_const_array_2d(self):
        jcuconst2d = cuda.jit('void(int32[:,:])')(cuconst2d)
        self.assertTrue('.const' in jcuconst2d.ptx)
        A = np.zeros_like(CONST2D, order='C')
        jcuconst2d[(2, 2), (5, 5)](A)
        self.assertTrue(np.all(A == CONST2D))

    def test_const_array_3d(self):
        jcuconst3d = cuda.jit('void(complex64[:,:,:])')(cuconst3d)
        self.assertTrue('.const' in jcuconst3d.ptx)
        A = np.zeros_like(CONST3D, order='F')
        jcuconst3d[1, (5, 5, 5)](A)
        self.assertTrue(np.all(A == CONST3D))

    def test_const_record(self):
        A = np.zeros(2, dtype=float)
        B = np.zeros(2, dtype=int)
        jcuconst = cuda.jit(cuconstRec).specialize(A, B)
        self.assertTrue('.const' in jcuconst.ptx)
        jcuconst[2, 1](A, B)
        np.testing.assert_allclose(A, CONST_RECORD['x'])
        np.testing.assert_allclose(B, CONST_RECORD['y'])

    def test_const_record_align(self):
        A = np.zeros(2, dtype=np.float64)
        B = np.zeros(2, dtype=np.float64)
        C = np.zeros(2, dtype=np.float64)
        D = np.zeros(2, dtype=np.float64)
        E = np.zeros(2, dtype=np.float64)
        jcuconst = cuda.jit(cuconstRecAlign).specialize(A, B, C, D, E)
        self.assertTrue('.const' in jcuconst.ptx)
        jcuconst[2, 1](A, B, C, D, E)
        np.testing.assert_allclose(A, CONST_RECORD_ALIGN['a'])
        np.testing.assert_allclose(B, CONST_RECORD_ALIGN['b'])
        np.testing.assert_allclose(C, CONST_RECORD_ALIGN['x'])
        np.testing.assert_allclose(D, CONST_RECORD_ALIGN['y'])
        np.testing.assert_allclose(E, CONST_RECORD_ALIGN['z'])


if __name__ == '__main__':
    unittest.main()
