import numpy as np

from numba import cuda, int32, complex128
from numba.cuda.testing import unittest, CUDATestCase


def culocal(A, B):
    C = cuda.local.array(1000, dtype=int32)
    for i in range(C.shape[0]):
        C[i] = A[i]
    for i in range(C.shape[0]):
        B[i] = C[i]


def culocalcomplex(A, B):
    C = cuda.local.array(100, dtype=complex128)
    for i in range(C.shape[0]):
        C[i] = A[i]
    for i in range(C.shape[0]):
        B[i] = C[i]


def culocal1tuple(A, B):
    C = cuda.local.array((5,), dtype=int32)
    for i in range(C.shape[0]):
        C[i] = A[i]
    for i in range(C.shape[0]):
        B[i] = C[i]


class TestCudaLocalMem(CUDATestCase):
    def test_local_array(self):
        jculocal = cuda.jit('void(int32[:], int32[:])')(culocal)
        self.assertTrue('.local' in jculocal.ptx)
        A = np.arange(1000, dtype='int32')
        B = np.zeros_like(A)
        jculocal[1, 1](A, B)
        self.assertTrue(np.all(A == B))

    def test_local_array_1_tuple(self):
        """Ensure that local arrays can be constructed with 1-tuple shape
        """
        jculocal = cuda.jit('void(int32[:], int32[:])')(culocal1tuple)
        # Don't check if .local is in the ptx because the optimizer
        # may reduce it to registers.
        A = np.arange(5, dtype='int32')
        B = np.zeros_like(A)
        jculocal[1, 1](A, B)
        self.assertTrue(np.all(A == B))

    def test_local_array_complex(self):
        sig = 'void(complex128[:], complex128[:])'
        jculocalcomplex = cuda.jit(sig)(culocalcomplex)
        # The local memory would be turned into register
        # self.assertTrue('.local' in jculocalcomplex.ptx)
        A = (np.arange(100, dtype='complex128') - 1) / 2j
        B = np.zeros_like(A)
        jculocalcomplex[1, 1](A, B)
        self.assertTrue(np.all(A == B))


if __name__ == '__main__':
    unittest.main()
