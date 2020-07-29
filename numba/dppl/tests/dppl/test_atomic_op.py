from __future__ import print_function, division, absolute_import

import numpy as np

import numba
from numba import dppl
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase
import dppl.ocldrv as ocldrv


def atomic_add(ary):
    tid = dppl.get_local_id(0)
    lm = dppl.local.static_alloc(32, numba.uint32)
    lm[tid] = 0
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    bin = ary[tid] % 32
    dppl.atomic.add(lm, bin, 1)
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    ary[tid] = lm[tid]


def atomic_add2(ary):
    tx = dppl.get_local_id(0)
    ty = dppl.get_local_id(1)
    lm = dppl.local.static_alloc((4, 8), numba.uint32)
    lm[tx, ty] = ary[tx, ty]
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    dppl.atomic.add(lm, (tx, ty), 1)
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    ary[tx, ty] = lm[tx, ty]


def atomic_add3(ary):
    tx = dppl.get_local_id(0)
    ty = dppl.get_local_id(1)
    lm = dppl.local.static_alloc((4, 8), numba.uint32)
    lm[tx, ty] = ary[tx, ty]
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    dppl.atomic.add(lm, (tx, numba.uint64(ty)), 1)
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    ary[tx, ty] = lm[tx, ty]



def call_fn_for_datatypes(fn, result, input, global_size):
    dtypes = [np.int32, np.int64, np.float32, np.double]

    for dtype in dtypes:
        a = np.array(input, dtype=dtype)

        with ocldrv.igpu_context(0) as device_env:
            if dtype == np.double and not device_env.device_support_float64_atomics():
                continue
            if dtype == np.int64 and not device_env.device_support_int64_atomics():
                continue
            fn[global_size, dppl.DEFAULT_LOCAL_SIZE](a)

        assert(a[0] == result)


@unittest.skipUnless(ocldrv.has_gpu_device, 'test only on GPU system')
class TestAtomicOp(DPPLTestCase):
    def test_atomic_add(self):
        @dppl.kernel
        def atomic_add(B):
            i = dppl.get_global_id(0)
            dppl.atomic.add(B, 0, 1)

        N = 100
        B = np.array([0])

        call_fn_for_datatypes(atomic_add, N, B, N)


    def test_atomic_sub(self):
        @dppl.kernel
        def atomic_sub(B):
            i = dppl.get_global_id(0)
            dppl.atomic.sub(B, 0, 1)

        N = 100
        B = np.array([100])

        call_fn_for_datatypes(atomic_sub, 0, B, N)


    def test_atomic_add1(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32)
        orig = ary.copy()

        dppl_atomic_add = dppl.kernel('void(uint32[:])')(atomic_add)
        with ocldrv.igpu_context(0) as device_env:
            dppl_atomic_add[32, dppl.DEFAULT_LOCAL_SIZE](ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(orig.size):
            gold[orig[i]] += 1

        self.assertTrue(np.all(ary == gold))

    def test_atomic_add2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        dppl_atomic_add2 = dppl.kernel('void(uint32[:,:])')(atomic_add2)
        with ocldrv.igpu_context(0) as device_env:
            dppl_atomic_add2[(4, 8), dppl.DEFAULT_LOCAL_SIZE](ary)
        self.assertTrue(np.all(ary == orig + 1))

    def test_atomic_add3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        dppl_atomic_add3 = dppl.kernel('void(uint32[:,:])')(atomic_add3)
        with ocldrv.igpu_context(0) as device_env:
            dppl_atomic_add3[(4, 8), dppl.DEFAULT_LOCAL_SIZE](ary)

        self.assertTrue(np.all(ary == orig + 1))


if __name__ == '__main__':
    unittest.main()
