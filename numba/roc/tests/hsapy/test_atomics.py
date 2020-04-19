import numpy as np

import numba
from numba import roc
import unittest


def atomic_add(ary):
    tid = roc.get_local_id(0)
    sm = roc.shared.array(32, numba.uint32)
    sm[tid] = 0
    roc.barrier(roc.CLK_GLOBAL_MEM_FENCE)
    bin = ary[tid] % 32
    roc.atomic.add(sm, bin, 1)
    roc.barrier(roc.CLK_GLOBAL_MEM_FENCE)
    ary[tid] = sm[tid]


def atomic_add2(ary):
    tx = roc.get_local_id(0)
    ty = roc.get_local_id(1)
    sm = roc.shared.array((4, 8), numba.uint32)
    sm[tx, ty] = ary[tx, ty]
    roc.barrier(roc.CLK_GLOBAL_MEM_FENCE)
    roc.atomic.add(sm, (tx, ty), 1)
    roc.barrier(roc.CLK_GLOBAL_MEM_FENCE)
    ary[tx, ty] = sm[tx, ty]


def atomic_add3(ary):
    tx = roc.get_local_id(0)
    ty = roc.get_local_id(1)
    sm = roc.shared.array((4, 8), numba.uint32)
    sm[tx, ty] = ary[tx, ty]
    roc.barrier(roc.CLK_GLOBAL_MEM_FENCE)
    roc.atomic.add(sm, (tx, numba.uint64(ty)), 1)
    roc.barrier(roc.CLK_GLOBAL_MEM_FENCE)
    ary[tx, ty] = sm[tx, ty]


class TestHsaAtomics(unittest.TestCase):
    def test_atomic_add(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32)
        orig = ary.copy()
        hsa_atomic_add = roc.jit('void(uint32[:])')(atomic_add)
        hsa_atomic_add[1, 32](ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(orig.size):
            gold[orig[i]] += 1

        self.assertTrue(np.all(ary == gold))

    def test_atomic_add2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        hsa_atomic_add2 = roc.jit('void(uint32[:,:])')(atomic_add2)
        hsa_atomic_add2[1, (4, 8)](ary)
        self.assertTrue(np.all(ary == orig + 1))

    def test_atomic_add3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        hsa_atomic_add3 = roc.jit('void(uint32[:,:])')(atomic_add3)
        hsa_atomic_add3[1, (4, 8)](ary)

        self.assertTrue(np.all(ary == orig + 1))

if __name__ == '__main__':
    unittest.main()
