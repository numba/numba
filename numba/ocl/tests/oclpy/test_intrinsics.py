from __future__ import print_function, absolute_import, division

import numpy as np
from numba import ocl, int32, float32
from numba.ocl.testing import unittest


def simple_threadidx(ary):
    i = ocl.threadIdx.x
    ary[0] = i


def fill_threadidx(ary):
    i = ocl.threadIdx.x
    ary[i] = i


def fill3d_threadidx(ary):
    i = ocl.threadIdx.x
    j = ocl.threadIdx.y
    k = ocl.threadIdx.z

    ary[i, j, k] = (i + 1) * (j + 1) * (k + 1)


def simple_grid1d(ary):
    i = ocl.grid(1)
    ary[i] = i


def simple_grid2d(ary):
    i, j = ocl.grid(2)
    ary[i, j] = i + j


def simple_gridsize1d(ary):
    i = ocl.grid(1)
    x = ocl.gridsize(1)
    if i == 0:
        ary[0] = x


def simple_gridsize2d(ary):
    i, j = ocl.grid(2)
    x, y = ocl.gridsize(2)
    if i == 0 and j == 0:
        ary[0] = x
        ary[1] = y


def intrinsic_forloop_step(c):
    startX, startY = ocl.grid(2)
    gridX = ocl.gridDim.x * ocl.blockDim.x
    gridY = ocl.gridDim.y * ocl.blockDim.y
    height, width = c.shape

    for x in range(startX, width, gridX):
        for y in range(startY, height, gridY):
            c[y, x] = x + y


class TestOclIntrinsic(unittest.TestCase):
    def test_simple_threadidx(self):
        compiled = ocl.jit("void(int32[:])")(simple_threadidx)
        ary = np.ones(1, dtype=np.int32)
        compiled(ary)
        self.assertTrue(ary[0] == 0)

    def test_fill_threadidx(self):
        compiled = ocl.jit("void(int32[:])")(fill_threadidx)
        N = 10
        ary = np.ones(N, dtype=np.int32)
        exp = np.arange(N, dtype=np.int32)
        compiled[1, N](ary)
        self.assertTrue(np.all(ary == exp))

    def test_fill3d_threadidx(self):
        X, Y, Z = 4, 5, 6

        def c_contigous():
            compiled = ocl.jit("void(int32[:,:,::1])")(fill3d_threadidx)
            ary = np.zeros((X, Y, Z), dtype=np.int32)
            compiled[1, (X, Y, Z)](ary)
            return ary

        def f_contigous():
            compiled = ocl.jit("void(int32[::1,:,:])")(fill3d_threadidx)
            ary = np.asfortranarray(np.zeros((X, Y, Z), dtype=np.int32))
            compiled[1, (X, Y, Z)](ary)
            return ary

        c_res = c_contigous()
        f_res = f_contigous()
        self.assertTrue(np.all(c_res == f_res))

    def test_simple_grid1d(self):
        compiled = ocl.jit("void(int32[::1])")(simple_grid1d)
        ntid, nctaid = 3, 7
        nelem = ntid * nctaid
        ary = np.empty(nelem, dtype=np.int32)
        compiled[nctaid, ntid](ary)
        self.assertTrue(np.all(ary == np.arange(nelem)))

    def test_simple_grid2d(self):
        compiled = ocl.jit("void(int32[:,::1])")(simple_grid2d)
        ntid = (4, 3)
        nctaid = (5, 6)
        shape = (ntid[0] * nctaid[0], ntid[1] * nctaid[1])
        ary = np.empty(shape, dtype=np.int32)
        exp = ary.copy()
        compiled[nctaid, ntid](ary)

        for i in range(ary.shape[0]):
            for j in range(ary.shape[1]):
                exp[i, j] = i + j

        self.assertTrue(np.all(ary == exp))

    def test_simple_gridsize1d(self):
        compiled = ocl.jit("void(int32[::1])")(simple_gridsize1d)
        ntid, nctaid = 3, 7
        ary = np.zeros(1, dtype=np.int32)
        compiled[nctaid, ntid](ary)
        self.assertEqual(ary[0], nctaid * ntid)

    def test_simple_gridsize2d(self):
        compiled = ocl.jit("void(int32[::1])")(simple_gridsize2d)
        ntid = (4, 3)
        nctaid = (5, 6)
        ary = np.zeros(2, dtype=np.int32)
        compiled[nctaid, ntid](ary)

        self.assertEqual(ary[0], nctaid[0] * ntid[0])
        self.assertEqual(ary[1], nctaid[1] * ntid[1])

    def test_intrinsic_forloop_step(self):
        compiled = ocl.jit("void(float32[:,::1])")(intrinsic_forloop_step)
        ntid = (4, 3)
        nctaid = (5, 6)
        shape = (ntid[0] * nctaid[0], ntid[1] * nctaid[1])
        ary = np.empty(shape, dtype=np.int32)

        compiled[nctaid, ntid](ary)

        gridX, gridY = shape
        height, width = ary.shape
        for i, j in zip(range(ntid[0]), range(ntid[1])):
            startX, startY = gridX + i, gridY + j
            for x in range(startX, width, gridX):
                for y in range(startY, height, gridY):
                    self.assertTrue(ary[y, x] == x + y, (ary[y, x], x + y))

    def test_3dgrid(self):
        @ocl.jit
        def foo(out):
            x, y, z = ocl.grid(3)
            a, b, c = ocl.gridsize(3)
            out[x, y, z] = a * b * c

        arr = np.zeros(9 ** 3, dtype=np.int32).reshape(9, 9, 9)
        foo[(3, 3, 3), (3, 3, 3)](arr)

        np.testing.assert_equal(arr, 9 ** 3)

    def test_3dgrid_2(self):
        @ocl.jit
        def foo(out):
            x, y, z = ocl.grid(3)
            a, b, c = ocl.gridsize(3)
            grid_is_right = (
                x == ocl.threadIdx.x + ocl.blockIdx.x * ocl.blockDim.x and
                y == ocl.threadIdx.y + ocl.blockIdx.y * ocl.blockDim.y and
                z == ocl.threadIdx.z + ocl.blockIdx.z * ocl.blockDim.z
            )
            gridsize_is_right = (a == ocl.blockDim.x * ocl.gridDim.x and
                                 b == ocl.blockDim.y * ocl.gridDim.y and
                                 c == ocl.blockDim.z * ocl.gridDim.z)
            out[x, y, z] = grid_is_right and gridsize_is_right

        x, y, z = (4 * 3, 3 * 2, 2 * 4)
        arr = np.zeros((x * y * z), dtype=np.bool).reshape(x, y, z)
        foo[(4, 3, 2), (3, 2, 4)](arr)

        self.assertTrue(np.all(arr))


if __name__ == '__main__':
    unittest.main()
