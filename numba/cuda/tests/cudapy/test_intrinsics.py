from __future__ import print_function, absolute_import, division

import numpy as np
import re
from numba import cuda, int32, float32
from numba.cuda.testing import unittest, SerialMixin, skip_on_cudasim


def simple_threadidx(ary):
    i = cuda.threadIdx.x
    ary[0] = i


def fill_threadidx(ary):
    i = cuda.threadIdx.x
    ary[i] = i


def fill3d_threadidx(ary):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    k = cuda.threadIdx.z

    ary[i, j, k] = (i + 1) * (j + 1) * (k + 1)


def simple_grid1d(ary):
    i = cuda.grid(1)
    ary[i] = i


def simple_grid2d(ary):
    i, j = cuda.grid(2)
    ary[i, j] = i + j


def simple_gridsize1d(ary):
    i = cuda.grid(1)
    x = cuda.gridsize(1)
    if i == 0:
        ary[0] = x


def simple_gridsize2d(ary):
    i, j = cuda.grid(2)
    x, y = cuda.gridsize(2)
    if i == 0 and j == 0:
        ary[0] = x
        ary[1] = y


def intrinsic_forloop_step(c):
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y
    height, width = c.shape

    for x in range(startX, width, gridX):
        for y in range(startY, height, gridY):
            c[y, x] = x + y


def simple_popc(ary, c):
    ary[0] = cuda.popc(c)


def simple_brev(ary, c):
    ary[0] = cuda.brev(c)


def simple_clz(ary, c):
    ary[0] = cuda.clz(c)


def simple_ffs(ary, c):
    ary[0] = cuda.ffs(c)


def branching_with_ifs(a, b, c):
    i = cuda.grid(1)

    if a[i] > 4:
        if b % 2 == 0:
            a[i] = c[i]
        else:
            a[i] = 13
    else:
        a[i] = 3


def branching_with_selps(a, b, c):
    i = cuda.grid(1)

    inner = cuda.selp(b % 2 == 0, c[i], 13)
    a[i] = cuda.selp(a[i] > 4, inner, 3)


def simple_laneid(ary):
    i = cuda.grid(1)
    ary[i] = cuda.laneid


def simple_warpsize(ary):
    ary[0] = cuda.warpsize


class TestCudaIntrinsic(SerialMixin, unittest.TestCase):
    def test_simple_threadidx(self):
        compiled = cuda.jit("void(int32[:])")(simple_threadidx)
        ary = np.ones(1, dtype=np.int32)
        compiled(ary)
        self.assertTrue(ary[0] == 0)

    def test_fill_threadidx(self):
        compiled = cuda.jit("void(int32[:])")(fill_threadidx)
        N = 10
        ary = np.ones(N, dtype=np.int32)
        exp = np.arange(N, dtype=np.int32)
        compiled[1, N](ary)
        self.assertTrue(np.all(ary == exp))

    def test_fill3d_threadidx(self):
        X, Y, Z = 4, 5, 6

        def c_contigous():
            compiled = cuda.jit("void(int32[:,:,::1])")(fill3d_threadidx)
            ary = np.zeros((X, Y, Z), dtype=np.int32)
            compiled[1, (X, Y, Z)](ary)
            return ary

        def f_contigous():
            compiled = cuda.jit("void(int32[::1,:,:])")(fill3d_threadidx)
            ary = np.asfortranarray(np.zeros((X, Y, Z), dtype=np.int32))
            compiled[1, (X, Y, Z)](ary)
            return ary

        c_res = c_contigous()
        f_res = f_contigous()
        self.assertTrue(np.all(c_res == f_res))

    def test_simple_grid1d(self):
        compiled = cuda.jit("void(int32[::1])")(simple_grid1d)
        ntid, nctaid = 3, 7
        nelem = ntid * nctaid
        ary = np.empty(nelem, dtype=np.int32)
        compiled[nctaid, ntid](ary)
        self.assertTrue(np.all(ary == np.arange(nelem)))

    def test_simple_grid2d(self):
        compiled = cuda.jit("void(int32[:,::1])")(simple_grid2d)
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
        compiled = cuda.jit("void(int32[::1])")(simple_gridsize1d)
        ntid, nctaid = 3, 7
        ary = np.zeros(1, dtype=np.int32)
        compiled[nctaid, ntid](ary)
        self.assertEqual(ary[0], nctaid * ntid)

    @skip_on_cudasim('Tests PTX emission')
    def test_selp(self):
        cu_branching_with_ifs = cuda.jit('void(i8[:], i8, i8[:])')(branching_with_ifs)
        cu_branching_with_selps = cuda.jit('void(i8[:], i8, i8[:])')(branching_with_selps)

        n = 32
        b = 6
        c = np.full(shape=32, fill_value=17, dtype=np.int64)

        expected = c.copy()
        expected[:5] = 3

        a = np.arange(n, dtype=np.int64)
        cu_branching_with_ifs[n, 1](a, b, c)
        ptx = cu_branching_with_ifs.inspect_asm()
        self.assertEqual(2, len(re.findall(r'\s+bra\s+', ptx)))
        np.testing.assert_array_equal(a, expected, err_msg='branching')

        a = np.arange(n, dtype=np.int64)
        cu_branching_with_selps[n, 1](a, b, c)
        ptx = cu_branching_with_selps.inspect_asm()
        self.assertEqual(0, len(re.findall(r'\s+bra\s+', ptx)))
        np.testing.assert_array_equal(a, expected, err_msg='selp')

    def test_simple_gridsize2d(self):
        compiled = cuda.jit("void(int32[::1])")(simple_gridsize2d)
        ntid = (4, 3)
        nctaid = (5, 6)
        ary = np.zeros(2, dtype=np.int32)
        compiled[nctaid, ntid](ary)

        self.assertEqual(ary[0], nctaid[0] * ntid[0])
        self.assertEqual(ary[1], nctaid[1] * ntid[1])

    def test_intrinsic_forloop_step(self):
        compiled = cuda.jit("void(float32[:,::1])")(intrinsic_forloop_step)
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
        @cuda.jit
        def foo(out):
            x, y, z = cuda.grid(3)
            a, b, c = cuda.gridsize(3)
            out[x, y, z] = a * b * c

        arr = np.zeros(9 ** 3, dtype=np.int32).reshape(9, 9, 9)
        foo[(3, 3, 3), (3, 3, 3)](arr)

        np.testing.assert_equal(arr, 9 ** 3)

    def test_3dgrid_2(self):
        @cuda.jit
        def foo(out):
            x, y, z = cuda.grid(3)
            a, b, c = cuda.gridsize(3)
            grid_is_right = (
                x == cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x and
                y == cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y and
                z == cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
            )
            gridsize_is_right = (a == cuda.blockDim.x * cuda.gridDim.x and
                                 b == cuda.blockDim.y * cuda.gridDim.y and
                                 c == cuda.blockDim.z * cuda.gridDim.z)
            out[x, y, z] = grid_is_right and gridsize_is_right

        x, y, z = (4 * 3, 3 * 2, 2 * 4)
        arr = np.zeros((x * y * z), dtype=np.bool).reshape(x, y, z)
        foo[(4, 3, 2), (3, 2, 4)](arr)

        self.assertTrue(np.all(arr))

    def test_popc_u4(self):
        compiled = cuda.jit("void(int32[:], uint32)")(simple_popc)
        ary = np.zeros(1, dtype=np.int32)
        compiled(ary, 0xF0)
        self.assertEquals(ary[0], 4)

    def test_popc_u8(self):
        compiled = cuda.jit("void(int32[:], uint64)")(simple_popc)
        ary = np.zeros(1, dtype=np.int32)
        compiled(ary, 0xF00000000000)
        self.assertEquals(ary[0], 4)

    def test_brev_u4(self):
        compiled = cuda.jit("void(uint32[:], uint32)")(simple_brev)
        ary = np.zeros(1, dtype=np.uint32)
        compiled(ary, 0x000030F0)
        self.assertEquals(ary[0], 0x0F0C0000)

    @skip_on_cudasim('only get given a Python "int", assumes 32 bits')
    def test_brev_u8(self):
        compiled = cuda.jit("void(uint64[:], uint64)")(simple_brev)
        ary = np.zeros(1, dtype=np.uint64)
        compiled(ary, 0x000030F0000030F0)
        self.assertEquals(ary[0], 0x0F0C00000F0C0000)

    def test_clz_i4(self):
        compiled = cuda.jit("void(int32[:], int32)")(simple_clz)
        ary = np.zeros(1, dtype=np.int32)
        compiled(ary, 0x00100000)
        self.assertEquals(ary[0], 11)

    def test_clz_u4(self):
        """
        Although the CUDA Math API (http://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html)
        only says int32 & int64 arguments are supported in C code, the LLVM
        IR input supports i8, i16, i32 & i64 (LLVM doesn't have a concept of
        unsigned integers, just unsigned operations on integers).
        http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics
        """
        compiled = cuda.jit("void(int32[:], uint32)")(simple_clz)
        ary = np.zeros(1, dtype=np.uint32)
        compiled(ary, 0x00100000)
        self.assertEquals(ary[0], 11)

    def test_clz_i4_1s(self):
        compiled = cuda.jit("void(int32[:], int32)")(simple_clz)
        ary = np.zeros(1, dtype=np.int32)
        compiled(ary, 0xFFFFFFFF)
        self.assertEquals(ary[0], 0)

    def test_clz_i4_0s(self):
        compiled = cuda.jit("void(int32[:], int32)")(simple_clz)
        ary = np.zeros(1, dtype=np.int32)
        compiled(ary, 0x0)
        self.assertEquals(ary[0], 32, "CUDA semantics")

    @skip_on_cudasim('only get given a Python "int", assumes 32 bits')
    def test_clz_i8(self):
        compiled = cuda.jit("void(int32[:], int64)")(simple_clz)
        ary = np.zeros(1, dtype=np.int32)
        compiled(ary, 0x000000000010000)
        self.assertEquals(ary[0], 47)

    def test_ffs_i4(self):
        compiled = cuda.jit("void(int32[:], int32)")(simple_ffs)
        ary = np.zeros(1, dtype=np.int32)
        compiled(ary, 0x00100000)
        self.assertEquals(ary[0], 20)

    def test_ffs_u4(self):
        compiled = cuda.jit("void(int32[:], uint32)")(simple_ffs)
        ary = np.zeros(1, dtype=np.uint32)
        compiled(ary, 0x00100000)
        self.assertEquals(ary[0], 20)

    def test_ffs_i4_1s(self):
        compiled = cuda.jit("void(int32[:], int32)")(simple_ffs)
        ary = np.zeros(1, dtype=np.int32)
        compiled(ary, 0xFFFFFFFF)
        self.assertEquals(ary[0], 0)

    def test_ffs_i4_0s(self):
        compiled = cuda.jit("void(int32[:], int32)")(simple_ffs)
        ary = np.zeros(1, dtype=np.int32)
        compiled(ary, 0x0)
        self.assertEquals(ary[0], 32, "CUDA semantics")

    @skip_on_cudasim('only get given a Python "int", assumes 32 bits')
    def test_ffs_i8(self):
        compiled = cuda.jit("void(int32[:], int64)")(simple_ffs)
        ary = np.zeros(1, dtype=np.int32)
        compiled(ary, 0x000000000010000)
        self.assertEquals(ary[0], 16)

    def test_simple_laneid(self):
        compiled = cuda.jit("void(int32[:])")(simple_laneid)
        count = 2
        ary = np.zeros(count*32, dtype=np.int32)
        exp = np.tile(np.arange(32, dtype=np.int32), count)
        compiled[1, count*32](ary)
        self.assertTrue(np.all(ary == exp))

    def test_simple_warpsize(self):
        compiled = cuda.jit("void(int32[:])")(simple_warpsize)
        ary = np.zeros(1, dtype=np.int32)
        compiled(ary)
        self.assertEquals(ary[0], 32, "CUDA semantics")


if __name__ == '__main__':
    unittest.main()
