from __future__ import print_function, division, absolute_import
import numpy as np

import numba
from numba import hsa
import numba.unittest_support as unittest


def atomic_add(ary):
    tid = hsa.get_local_id(0)
    sm = hsa.shared.array(32, numba.uint32)
    sm[tid] = 0
    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)
    bin = ary[tid] % 32
    hsa.atomic.add(sm, bin, 1)
    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)
    ary[tid] = sm[tid]


def atomic_add2(ary):
    tx = hsa.get_local_id(0)
    ty = hsa.get_local_id(1)
    sm = hsa.shared.array((4, 8), numba.uint32)
    sm[tx, ty] = ary[tx, ty]
    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)
    hsa.atomic.add(sm, (tx, ty), 1)
    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)
    ary[tx, ty] = sm[tx, ty]


def atomic_add3(ary):
    tx = hsa.get_local_id(0)
    ty = hsa.get_local_id(1)
    sm = hsa.shared.array((4, 8), numba.uint32)
    sm[tx, ty] = ary[tx, ty]
    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)
    hsa.atomic.add(sm, (tx, numba.uint64(ty)), 1)
    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)
    ary[tx, ty] = sm[tx, ty]


class TestHsaAtomics(unittest.TestCase):
    def test_atomic_add(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32)
        orig = ary.copy()
        hsa_atomic_add = hsa.jit('void(uint32[:])')(atomic_add)
        hsa_atomic_add[1, 32](ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(orig.size):
            gold[orig[i]] += 1

        self.assertTrue(np.all(ary == gold))

    def test_atomic_add2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        hsa_atomic_add2 = hsa.jit('void(uint32[:,:])')(atomic_add2)
        hsa_atomic_add2[1, (4, 8)](ary)
        self.assertTrue(np.all(ary == orig + 1))

    def test_atomic_add3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        hsa_atomic_add3 = hsa.jit('void(uint32[:,:])')(atomic_add3)
        hsa_atomic_add3[1, (4, 8)](ary)

        self.assertTrue(np.all(ary == orig + 1))

if __name__ == '__main__':
    unittest.main()
