from __future__ import print_function, division, absolute_import
import numpy as np
from numba import cuda, uint32, uint64, float32
from numba.cuda.testing import unittest


def atomic_add(ary):
    tid = cuda.threadIdx.x
    sm = cuda.shared.array(32, uint32)
    sm[tid] = 0
    cuda.syncthreads()
    bin = ary[tid] % 32
    cuda.atomic.add(sm, bin, 1)
    cuda.syncthreads()
    ary[tid] = sm[tid]


def atomic_add2(ary):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    sm = cuda.shared.array((4, 8), uint32)
    sm[tx, ty] = ary[tx, ty]
    cuda.syncthreads()
    cuda.atomic.add(sm, (tx, ty), 1)
    cuda.syncthreads()
    ary[tx, ty] = sm[tx, ty]


def atomic_add3(ary):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    sm = cuda.shared.array((4, 8), uint32)
    sm[tx, ty] = ary[tx, ty]
    cuda.syncthreads()
    cuda.atomic.add(sm, (tx, uint64(ty)), 1)
    cuda.syncthreads()
    ary[tx, ty] = sm[tx, ty]


def atomic_add_float(ary):
    tid = cuda.threadIdx.x
    sm = cuda.shared.array(32, float32)
    sm[tid] = 0
    cuda.syncthreads()
    bin = ary[tid] % 32
    cuda.atomic.add(sm, bin, 1.0)
    cuda.syncthreads()
    ary[tid] = sm[tid]


def atomic_add_float_2(ary):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    sm = cuda.shared.array((4, 8), float32)
    sm[tx, ty] = ary[tx, ty]
    cuda.syncthreads()
    cuda.atomic.add(sm, (tx, ty), 1)
    cuda.syncthreads()
    ary[tx, ty] = sm[tx, ty]


def atomic_add_float_3(ary):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    sm = cuda.shared.array((4, 8), float32)
    sm[tx, ty] = ary[tx, ty]
    cuda.syncthreads()
    cuda.atomic.add(sm, (tx, uint64(ty)), 1)
    cuda.syncthreads()
    ary[tx, ty] = sm[tx, ty]


class TestCudaAtomics(unittest.TestCase):
    def test_atomic_add(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32)
        orig = ary.copy()
        cuda_atomic_add = cuda.jit('void(uint32[:])')(atomic_add)
        cuda_atomic_add[1, 32](ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(orig.size):
            gold[orig[i]] += 1

        self.assertTrue(np.all(ary == gold))

    def test_atomic_add2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        cuda_atomic_add2 = cuda.jit('void(uint32[:,:])')(atomic_add2)
        cuda_atomic_add2[1, (4, 8)](ary)
        self.assertTrue(np.all(ary == orig + 1))

    def test_atomic_add3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        cuda_atomic_add3 = cuda.jit('void(uint32[:,:])')(atomic_add3)
        cuda_atomic_add3[1, (4, 8)](ary)

        self.assertTrue(np.all(ary == orig + 1))

    def test_atomic_add_float(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float32)
        orig = ary.copy()
        cuda_atomic_add_float = cuda.jit('void(float32[:])')(atomic_add_float)
        cuda_atomic_add_float[1, 32](ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(orig.size):
            gold[orig[i]] += 1.0

        self.assertTrue(np.all(ary == gold))

    def test_atomic_add_float_2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float32).reshape(4, 8)
        orig = ary.copy()
        cuda_atomic_add2 = cuda.jit('void(float32[:,:])')(atomic_add_float_2)
        cuda_atomic_add2[1, (4, 8)](ary)
        self.assertTrue(np.all(ary == orig + 1))

    def test_atomic_add_float_3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float32).reshape(4, 8)
        orig = ary.copy()
        cuda_atomic_add3 = cuda.jit('void(float32[:,:])')(atomic_add_float_3)
        cuda_atomic_add3[1, (4, 8)](ary)

        self.assertTrue(np.all(ary == orig + 1))


if __name__ == '__main__':
    unittest.main()
