from __future__ import print_function, division, absolute_import

import random
import numpy as np

from numba import config
from numba import ocl, uint32, uint64, float32, float64
from numba.ocl.testing import unittest


def cc_32_or_above():
    return ocl.current_context().device.compute_capability >= (3, 2)


def skip_unless_cc_32(fn):
    return unittest.skipUnless(cc_32_or_above(), "require cc >= 3.2")(fn)


def atomic_add(ary):
    tid = ocl.threadIdx.x
    sm = ocl.shared.array(32, uint32)
    sm[tid] = 0
    ocl.syncthreads()
    bin = ary[tid] % 32
    ocl.atomic.add(sm, bin, 1)
    ocl.syncthreads()
    ary[tid] = sm[tid]


def atomic_add2(ary):
    tx = ocl.threadIdx.x
    ty = ocl.threadIdx.y
    sm = ocl.shared.array((4, 8), uint32)
    sm[tx, ty] = ary[tx, ty]
    ocl.syncthreads()
    ocl.atomic.add(sm, (tx, ty), 1)
    ocl.syncthreads()
    ary[tx, ty] = sm[tx, ty]


def atomic_add3(ary):
    tx = ocl.threadIdx.x
    ty = ocl.threadIdx.y
    sm = ocl.shared.array((4, 8), uint32)
    sm[tx, ty] = ary[tx, ty]
    ocl.syncthreads()
    ocl.atomic.add(sm, (tx, uint64(ty)), 1)
    ocl.syncthreads()
    ary[tx, ty] = sm[tx, ty]


def atomic_add_float(ary):
    tid = ocl.threadIdx.x
    sm = ocl.shared.array(32, float32)
    sm[tid] = 0
    ocl.syncthreads()
    bin = int(ary[tid] % 32)
    ocl.atomic.add(sm, bin, 1.0)
    ocl.syncthreads()
    ary[tid] = sm[tid]


def atomic_add_float_2(ary):
    tx = ocl.threadIdx.x
    ty = ocl.threadIdx.y
    sm = ocl.shared.array((4, 8), float32)
    sm[tx, ty] = ary[tx, ty]
    ocl.syncthreads()
    ocl.atomic.add(sm, (tx, ty), 1)
    ocl.syncthreads()
    ary[tx, ty] = sm[tx, ty]


def atomic_add_float_3(ary):
    tx = ocl.threadIdx.x
    ty = ocl.threadIdx.y
    sm = ocl.shared.array((4, 8), float32)
    sm[tx, ty] = ary[tx, ty]
    ocl.syncthreads()
    ocl.atomic.add(sm, (tx, uint64(ty)), 1)
    ocl.syncthreads()
    ary[tx, ty] = sm[tx, ty]


def atomic_add_double_global(idx, ary):
    tid = ocl.threadIdx.x
    bin = idx[tid] % 32
    ocl.atomic.add(ary, bin, 1.0)


def atomic_add_double_global_2(ary):
    tx = ocl.threadIdx.x
    ty = ocl.threadIdx.y
    ocl.atomic.add(ary, (tx, ty), 1)


def atomic_add_double_global_3(ary):
    tx = ocl.threadIdx.x
    ty = ocl.threadIdx.y
    ocl.atomic.add(ary, (tx, uint64(ty)), 1)


def atomic_add_double(idx, ary):
    tid = ocl.threadIdx.x
    sm = ocl.shared.array(32, float64)
    sm[tid] = 0.0
    ocl.syncthreads()
    bin = idx[tid] % 32
    ocl.atomic.add(sm, bin, 1.0)
    ocl.syncthreads()
    ary[tid] = sm[tid]


def atomic_add_double_2(ary):
    tx = ocl.threadIdx.x
    ty = ocl.threadIdx.y
    sm = ocl.shared.array((4, 8), float64)
    sm[tx, ty] = ary[tx, ty]
    ocl.syncthreads()
    ocl.atomic.add(sm, (tx, ty), 1)
    ocl.syncthreads()
    ary[tx, ty] = sm[tx, ty]


def atomic_add_double_3(ary):
    tx = ocl.threadIdx.x
    ty = ocl.threadIdx.y
    sm = ocl.shared.array((4, 8), float64)
    sm[tx, ty] = ary[tx, ty]
    ocl.syncthreads()
    ocl.atomic.add(sm, (tx, uint64(ty)), 1)
    ocl.syncthreads()
    ary[tx, ty] = sm[tx, ty]


def atomic_max(res, ary):
    tx = ocl.threadIdx.x
    bx = ocl.blockIdx.x
    ocl.atomic.max(res, 0, ary[tx, bx])


def atomic_min(res, ary):
    tx = ocl.threadIdx.x
    bx = ocl.blockIdx.x
    ocl.atomic.min(res, 0, ary[tx, bx])


def atomic_max_double_normalizedindex(res, ary):
    tx = ocl.threadIdx.x
    bx = ocl.blockIdx.x
    ocl.atomic.max(res, 0, ary[tx, uint64(bx)])


def atomic_max_double_oneindex(res, ary):
    tx = ocl.threadIdx.x
    ocl.atomic.max(res, 0, ary[tx])


def atomic_max_double_shared(res, ary):
    tid = ocl.threadIdx.x
    smary = ocl.shared.array(32, float64)
    smary[tid] = ary[tid]
    smres = ocl.shared.array(1, float64)
    if tid == 0:
        smres[0] = res[0]
    ocl.syncthreads()
    ocl.atomic.max(smres, 0, smary[tid])
    ocl.syncthreads()
    if tid == 0:
        res[0] = smres[0]


def atomic_compare_and_swap(res, old, ary):
    gid = ocl.grid(1)
    if gid < res.size:
        out = ocl.atomic.compare_and_swap(res[gid:], -99, ary[gid])
        old[gid] = out


class TestOclAtomics(unittest.TestCase):
    def test_atomic_add(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32)
        orig = ary.copy()
        ocl_atomic_add = ocl.jit('void(uint32[:])')(atomic_add)
        ocl_atomic_add[1, 32](ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(orig.size):
            gold[orig[i]] += 1

        self.assertTrue(np.all(ary == gold))

    def test_atomic_add2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        ocl_atomic_add2 = ocl.jit('void(uint32[:,:])')(atomic_add2)
        ocl_atomic_add2[1, (4, 8)](ary)
        self.assertTrue(np.all(ary == orig + 1))

    def test_atomic_add3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        ocl_atomic_add3 = ocl.jit('void(uint32[:,:])')(atomic_add3)
        ocl_atomic_add3[1, (4, 8)](ary)

        self.assertTrue(np.all(ary == orig + 1))

    def test_atomic_add_float(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float32)
        orig = ary.copy().astype(np.intp)
        ocl_atomic_add_float = ocl.jit('void(float32[:])')(atomic_add_float)
        ocl_atomic_add_float[1, 32](ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(orig.size):
            gold[orig[i]] += 1.0

        self.assertTrue(np.all(ary == gold))

    def test_atomic_add_float_2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float32).reshape(4, 8)
        orig = ary.copy()
        ocl_atomic_add2 = ocl.jit('void(float32[:,:])')(atomic_add_float_2)
        ocl_atomic_add2[1, (4, 8)](ary)
        self.assertTrue(np.all(ary == orig + 1))

    def test_atomic_add_float_3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float32).reshape(4, 8)
        orig = ary.copy()
        ocl_atomic_add3 = ocl.jit('void(float32[:,:])')(atomic_add_float_3)
        ocl_atomic_add3[1, (4, 8)](ary)

        self.assertTrue(np.all(ary == orig + 1))

    def test_atomic_add_double(self):
        idx = np.random.randint(0, 32, size=32)
        ary = np.zeros(32, np.float64)
        ocl_func = ocl.jit('void(int64[:], float64[:])')(atomic_add_double)
        ocl_func[1, 32](idx, ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(idx.size):
            gold[idx[i]] += 1.0

        np.testing.assert_equal(ary, gold)

    def test_atomic_add_double_2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float64).reshape(4, 8)
        orig = ary.copy()
        ocl_func = ocl.jit('void(float64[:,:])')(atomic_add_double_2)
        ocl_func[1, (4, 8)](ary)
        np.testing.assert_equal(ary, orig + 1)

    def test_atomic_add_double_3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float64).reshape(4, 8)
        orig = ary.copy()
        ocl_func = ocl.jit('void(float64[:,:])')(atomic_add_double_3)
        ocl_func[1, (4, 8)](ary)

        np.testing.assert_equal(ary, orig + 1)

    def test_atomic_add_double_global(self):
        idx = np.random.randint(0, 32, size=32)
        ary = np.zeros(32, np.float64)
        ocl_func = ocl.jit('void(int64[:], float64[:])')(atomic_add_double_global)
        ocl_func[1, 32](idx, ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(idx.size):
            gold[idx[i]] += 1.0

        np.testing.assert_equal(ary, gold)

    def test_atomic_add_double_global_2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float64).reshape(4, 8)
        orig = ary.copy()
        ocl_func = ocl.jit('void(float64[:,:])')(atomic_add_double_global_2)
        ocl_func[1, (4, 8)](ary)
        np.testing.assert_equal(ary, orig + 1)

    def test_atomic_add_double_global_3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float64).reshape(4, 8)
        orig = ary.copy()
        ocl_func = ocl.jit('void(float64[:,:])')(atomic_add_double_global_3)
        ocl_func[1, (4, 8)](ary)

        np.testing.assert_equal(ary, orig + 1)

    def check_atomic_max(self, dtype, lo, hi):
        vals = np.random.randint(lo, hi, size=(32, 32)).astype(dtype)
        res = np.zeros(1, dtype=vals.dtype)
        ocl_func = ocl.jit(atomic_max)
        ocl_func[32, 32](res, vals)
        gold = np.max(vals)
        np.testing.assert_equal(res, gold)

    def test_atomic_max_int32(self):
        self.check_atomic_max(dtype=np.int32, lo=-65535, hi=65535)

    def test_atomic_max_uint32(self):
        self.check_atomic_max(dtype=np.uint32, lo=0, hi=65535)

    @skip_unless_cc_32
    def test_atomic_max_int64(self):
        self.check_atomic_max(dtype=np.int64, lo=-65535, hi=65535)

    @skip_unless_cc_32
    def test_atomic_max_uint64(self):
        self.check_atomic_max(dtype=np.uint64, lo=0, hi=65535)

    def test_atomic_max_float32(self):
        self.check_atomic_max(dtype=np.float32, lo=-65535, hi=65535)

    def test_atomic_max_double(self):
        self.check_atomic_max(dtype=np.float64, lo=-65535, hi=65535)

    def check_atomic_min(self, dtype, lo, hi):
        vals = np.random.randint(lo, hi, size=(32, 32)).astype(dtype)
        res = np.array([65535], dtype=vals.dtype)
        ocl_func = ocl.jit(atomic_min)
        ocl_func[32, 32](res, vals)

        gold = np.min(vals)
        np.testing.assert_equal(res, gold)

    def test_atomic_min_int32(self):
        self.check_atomic_min(dtype=np.int32, lo=-65535, hi=65535)

    def test_atomic_min_uint32(self):
        self.check_atomic_min(dtype=np.uint32, lo=0, hi=65535)

    @skip_unless_cc_32
    def test_atomic_min_int64(self):
        self.check_atomic_min(dtype=np.int64, lo=-65535, hi=65535)

    @skip_unless_cc_32
    def test_atomic_min_uint64(self):
        self.check_atomic_min(dtype=np.uint64, lo=0, hi=65535)

    def test_atomic_min_float(self):
        self.check_atomic_min(dtype=np.float32, lo=-65535, hi=65535)

    def test_atomic_min_double(self):
        self.check_atomic_min(dtype=np.float64, lo=-65535, hi=65535)

    def test_atomic_max_double_normalizedindex(self):
        vals = np.random.randint(0, 65535, size=(32, 32)).astype(np.float64)
        res = np.zeros(1, np.float64)
        ocl_func = ocl.jit('void(float64[:], float64[:,:])')(
            atomic_max_double_normalizedindex)
        ocl_func[32, 32](res, vals)

        gold = np.max(vals)
        np.testing.assert_equal(res, gold)

    def test_atomic_max_double_oneindex(self):
        vals = np.random.randint(0, 128, size=32).astype(np.float64)
        res = np.zeros(1, np.float64)
        ocl_func = ocl.jit('void(float64[:], float64[:])')(
            atomic_max_double_oneindex)
        ocl_func[1, 32](res, vals)

        gold = np.max(vals)
        np.testing.assert_equal(res, gold)

    def test_atomic_max_nan_location(self):
        vals = np.random.randint(0, 128, size=(1,1)).astype(np.float64)
        gold = vals.copy().reshape(1)
        res = np.zeros(1, np.float64) + np.nan
        ocl_func = ocl.jit('void(float64[:], float64[:,:])')(atomic_max)
        ocl_func[1, 1](res, vals)

        np.testing.assert_equal(res, gold)

    def test_atomic_max_nan_val(self):
        res = np.random.randint(0, 128, size=1).astype(np.float64)
        gold = res.copy()
        vals = np.zeros((1, 1), np.float64) + np.nan
        ocl_func = ocl.jit('void(float64[:], float64[:,:])')(atomic_max)
        ocl_func[1, 1](res, vals)

        np.testing.assert_equal(res, gold)

    def test_atomic_max_double_shared(self):
        vals = np.random.randint(0, 32, size=32).astype(np.float64)
        res = np.zeros(1, np.float64)
        ocl_func = ocl.jit('void(float64[:], float64[:])')(atomic_max_double_shared)
        ocl_func[1, 32](res, vals)

        gold = np.max(vals)
        np.testing.assert_equal(res, gold)

    def test_atomic_compare_and_swap(self):
        n = 100
        res = [-99] * (n // 2) + [-1] * (n // 2)
        random.shuffle(res)
        res = np.asarray(res, dtype=np.int32)
        out = np.zeros_like(res)
        ary = np.random.randint(1, 10, size=res.size).astype(res.dtype)

        fill_mask = res == -99
        unfill_mask = res == -1

        expect_res = np.zeros_like(res)
        expect_res[fill_mask] = ary[fill_mask]
        expect_res[unfill_mask] = -1

        expect_out = np.zeros_like(out)
        expect_out[fill_mask] = res[fill_mask]
        expect_out[unfill_mask] = -1

        ocl_func = ocl.jit(atomic_compare_and_swap)
        ocl_func[10, 10](res, out, ary)

        np.testing.assert_array_equal(expect_res, res)
        np.testing.assert_array_equal(expect_out, out)


if __name__ == '__main__':
    unittest.main()
