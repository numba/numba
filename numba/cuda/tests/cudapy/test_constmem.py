import numpy as np

from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM

CONST_EMPTY = np.array([])
CONST1D = np.arange(10, dtype=np.float64) / 2.
CONST2D = np.asfortranarray(
    np.arange(100, dtype=np.int32).reshape(10, 10))
CONST3D = ((np.arange(5 * 5 * 5, dtype=np.complex64).reshape(5, 5, 5) + 1j) /
           2j)
CONST3BYTES = np.arange(3, dtype=np.uint8)

CONST_RECORD_EMPTY = np.array(
    [],
    dtype=[('x', float), ('y', int)])
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


def cuconstEmpty(A):
    C = cuda.const.array_like(CONST_EMPTY)
    i = cuda.grid(1)
    A[i] = len(C)


def cuconst(A):
    C = cuda.const.array_like(CONST1D)
    i = cuda.grid(1)

    # +1 or it'll be loaded & stored as a u32
    A[i] = C[i] + 1.0


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


def cuconstRecEmpty(A):
    C = cuda.const.array_like(CONST_RECORD_EMPTY)
    i = cuda.grid(1)
    A[i] = len(C)


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


def cuconstAlign(z):
    a = cuda.const.array_like(CONST3BYTES)
    b = cuda.const.array_like(CONST1D)
    i = cuda.grid(1)
    z[i] = a[i] + b[i]


class TestCudaConstantMemory(CUDATestCase):
    def test_const_array(self):
        jcuconst = cuda.jit('void(float64[:])')(cuconst)
        A = np.zeros_like(CONST1D)
        jcuconst[2, 5](A)
        self.assertTrue(np.all(A == CONST1D + 1))

        if not ENABLE_CUDASIM:
            self.assertIn(
                'ld.const.f64',
                jcuconst.ptx,
                "as we're adding to it, load as a double")

    def test_const_empty(self):
        jcuconstEmpty = cuda.jit('void(float64[:])')(cuconstEmpty)
        A = np.full(1, fill_value=-1, dtype=int)
        jcuconstEmpty[1, 1](A)
        self.assertTrue(np.all(A == 0))

    def test_const_align(self):
        jcuconstAlign = cuda.jit('void(float64[:])')(cuconstAlign)
        A = np.full(3, fill_value=np.nan, dtype=float)
        jcuconstAlign[1, 3](A)
        self.assertTrue(np.all(A == (CONST3BYTES + CONST1D[:3])))

    def test_const_array_2d(self):
        jcuconst2d = cuda.jit('void(int32[:,:])')(cuconst2d)
        A = np.zeros_like(CONST2D, order='C')
        jcuconst2d[(2, 2), (5, 5)](A)
        self.assertTrue(np.all(A == CONST2D))

        if not ENABLE_CUDASIM:
            self.assertIn(
                'ld.const.u32',
                jcuconst2d.ptx,
                "load the ints as ints")

    def test_const_array_3d(self):
        jcuconst3d = cuda.jit('void(complex64[:,:,:])')(cuconst3d)
        A = np.zeros_like(CONST3D, order='F')
        jcuconst3d[1, (5, 5, 5)](A)
        self.assertTrue(np.all(A == CONST3D))

        if not ENABLE_CUDASIM:
            self.assertIn(
                'ld.const.v2.u32',
                jcuconst3d.ptx,
                "load the two halves of the complex as u32s")

    def test_const_record_empty(self):
        jcuconstRecEmpty = cuda.jit('void(float64[:])')(cuconstRecEmpty)
        A = np.full(1, fill_value=-1, dtype=int)
        jcuconstRecEmpty[1, 1](A)
        self.assertTrue(np.all(A == 0))

    def test_const_record(self):
        A = np.zeros(2, dtype=float)
        B = np.zeros(2, dtype=int)
        jcuconst = cuda.jit(cuconstRec).specialize(A, B)

        if not ENABLE_CUDASIM:
            if not any(c in jcuconst.ptx for c in [
                # a vector load: the compiler fuses the load
                # of the x and y fields into a single instruction!
                'ld.const.v2.u64',

                # for some reason Win64 / Py3 / CUDA 9.1 decides
                # to do two u32 loads, and shifts and ors the
                # values to get the float `x` field, then uses
                # another ld.const.u32 to load the int `y` as
                # a 32-bit value!
                'ld.const.u32',
            ]):
                raise AssertionError(
                    "the compiler should realise it doesn't " \
                    "need to interpret the bytes as float!")

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

        if not ENABLE_CUDASIM:
            self.assertIn(
                'ld.const.v4.u8',
                jcuconst.ptx,
                'load the first three bytes as a vector')

            self.assertIn(
                'ld.const.u32',
                jcuconst.ptx,
                'load the uint32 natively')

            self.assertIn(
                'ld.const.u8',
                jcuconst.ptx,
                'load the last byte by itself')

        jcuconst[2, 1](A, B, C, D, E)
        np.testing.assert_allclose(A, CONST_RECORD_ALIGN['a'])
        np.testing.assert_allclose(B, CONST_RECORD_ALIGN['b'])
        np.testing.assert_allclose(C, CONST_RECORD_ALIGN['x'])
        np.testing.assert_allclose(D, CONST_RECORD_ALIGN['y'])
        np.testing.assert_allclose(E, CONST_RECORD_ALIGN['z'])


if __name__ == '__main__':
    unittest.main()
