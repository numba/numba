from __future__ import print_function, division, absolute_import

import numpy as np

import numba
from numba import dppl
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase
import dpctl

def atomic_add_int32(ary):
    tid = dppl.get_local_id(0)
    lm = dppl.local.static_alloc(32, numba.uint32)
    lm[tid] = 0
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    bin = ary[tid] % 32
    dppl.atomic.add(lm, bin, 1)
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    ary[tid] = lm[tid]


def atomic_sub_int32(ary):
    tid = dppl.get_local_id(0)
    lm = dppl.local.static_alloc(32, numba.uint32)
    lm[tid] = 0
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    bin = ary[tid] % 32
    dppl.atomic.sub(lm, bin, 1)
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    ary[tid] = lm[tid]


def atomic_add_float32(ary):
    lm = dppl.local.static_alloc(1, numba.float32)
    lm[0] = ary[0]
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    dppl.atomic.add(lm, 0, 1)
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    ary[0] = lm[0]


def atomic_sub_float32(ary):
    lm = dppl.local.static_alloc(1, numba.float32)
    lm[0] = ary[0]
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    dppl.atomic.sub(lm, 0, 1)
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    ary[0] = lm[0]


def atomic_add_int64(ary):
    lm = dppl.local.static_alloc(1, numba.int64)
    lm[0] = ary[0]
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    dppl.atomic.add(lm, 0, 1)
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    ary[0] = lm[0]


def atomic_sub_int64(ary):
    lm = dppl.local.static_alloc(1, numba.int64)
    lm[0] = ary[0]
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    dppl.atomic.sub(lm, 0, 1)
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    ary[0] = lm[0]


def atomic_add_float64(ary):
    lm = dppl.local.static_alloc(1, numba.float64)
    lm[0] = ary[0]
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    dppl.atomic.add(lm, 0, 1)
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    ary[0] = lm[0]


def atomic_sub_float64(ary):
    lm = dppl.local.static_alloc(1, numba.float64)
    lm[0] = ary[0]
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    dppl.atomic.sub(lm, 0, 1)
    dppl.barrier(dppl.CLK_GLOBAL_MEM_FENCE)
    ary[0] = lm[0]


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

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            # TODO: dpctl needs to expose this functions
            #if dtype == np.double and not device_env.device_support_float64_atomics():
            #    continue
            #if dtype == np.int64 and not device_env.device_support_int64_atomics():
            #    continue
            fn[global_size, dppl.DEFAULT_LOCAL_SIZE](a)

        assert(a[0] == result)


@unittest.skipUnless(dpctl.has_gpu_queues(), 'test only on GPU system')
@unittest.skipUnless(numba.dppl.ocl.atomic_support_present(), 'test only when atomic support is present')
class TestAtomicOp(DPPLTestCase):
    def test_atomic_add_global(self):
        @dppl.kernel
        def atomic_add(B):
            dppl.atomic.add(B, 0, 1)

        N = 100
        B = np.array([0])

        call_fn_for_datatypes(atomic_add, N, B, N)


    def test_atomic_sub_global(self):
        @dppl.kernel
        def atomic_sub(B):
            dppl.atomic.sub(B, 0, 1)

        N = 100
        B = np.array([100])

        call_fn_for_datatypes(atomic_sub, 0, B, N)


    def test_atomic_add_local_int32(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32)
        orig = ary.copy()

        #dppl_atomic_add = dppl.kernel('void(uint32[:])')(atomic_add_int32)
        dppl_atomic_add = dppl.kernel(atomic_add_int32)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            dppl_atomic_add[32, dppl.DEFAULT_LOCAL_SIZE](ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(orig.size):
            gold[orig[i]] += 1

        self.assertTrue(np.all(ary == gold))


    def test_atomic_sub_local_int32(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32)
        orig = ary.copy()

        #dppl_atomic_sub = dppl.kernel('void(uint32[:])')(atomic_sub_int32)
        dppl_atomic_sub = dppl.kernel(atomic_sub_int32)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            dppl_atomic_sub[32, dppl.DEFAULT_LOCAL_SIZE](ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(orig.size):
            gold[orig[i]] -= 1

        self.assertTrue(np.all(ary == gold))


    def test_atomic_add_local_float32(self):
        ary = np.array([0], dtype=np.float32)

        #dppl_atomic_add = dppl.kernel('void(float32[:])')(atomic_add_float32)
        dppl_atomic_add = dppl.kernel(atomic_add_float32)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            dppl_atomic_add[32, dppl.DEFAULT_LOCAL_SIZE](ary)

        self.assertTrue(ary[0] == 32)


    def test_atomic_sub_local_float32(self):
        ary = np.array([32], dtype=np.float32)

        #dppl_atomic_sub = dppl.kernel('void(float32[:])')(atomic_sub_float32)
        dppl_atomic_sub = dppl.kernel(atomic_sub_float32)
        with dpctl.device_context("opencl:gpu") as gpu_queue:

            dppl_atomic_sub[32, dppl.DEFAULT_LOCAL_SIZE](ary)

        self.assertTrue(ary[0] == 0)


    def test_atomic_add_local_int64(self):
        ary = np.array([0], dtype=np.int64)

        #dppl_atomic_add = dppl.kernel('void(int64[:])')(atomic_add_int64)
        dppl_atomic_add = dppl.kernel(atomic_add_int64)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            # TODO: dpctl needs to expose this functions
            #if device_env.device_support_int64_atomics():
            dppl_atomic_add[32, dppl.DEFAULT_LOCAL_SIZE](ary)
            self.assertTrue(ary[0] == 32)
            #else:
            #    return


    def test_atomic_sub_local_int64(self):
        ary = np.array([32], dtype=np.int64)

        #fn = dppl.kernel('void(int64[:])')(atomic_sub_int64)
        fn = dppl.kernel(atomic_sub_int64)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            # TODO: dpctl needs to expose this functions
            #if device_env.device_support_int64_atomics():
            fn[32, dppl.DEFAULT_LOCAL_SIZE](ary)
            self.assertTrue(ary[0] == 0)
            #else:
            #    return


    def test_atomic_add_local_float64(self):
        ary = np.array([0], dtype=np.double)

        #fn = dppl.kernel('void(float64[:])')(atomic_add_float64)
        fn = dppl.kernel(atomic_add_float64)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            # TODO: dpctl needs to expose this functions
            #if device_env.device_support_float64_atomics():
            fn[32, dppl.DEFAULT_LOCAL_SIZE](ary)
            self.assertTrue(ary[0] == 32)
            #else:
            #    return


    def test_atomic_sub_local_float64(self):
        ary = np.array([32], dtype=np.double)

        #fn = dppl.kernel('void(float64[:])')(atomic_sub_int64)
        fn = dppl.kernel(atomic_sub_int64)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            # TODO: dpctl needs to expose this functions
            #if device_env.device_support_float64_atomics():
            fn[32, dppl.DEFAULT_LOCAL_SIZE](ary)
            self.assertTrue(ary[0] == 0)
            #else:
            #    return


    def test_atomic_add2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        #dppl_atomic_add2 = dppl.kernel('void(uint32[:,:])')(atomic_add2)
        dppl_atomic_add2 = dppl.kernel(atomic_add2)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            dppl_atomic_add2[(4, 8), dppl.DEFAULT_LOCAL_SIZE](ary)
        self.assertTrue(np.all(ary == orig + 1))


    def test_atomic_add3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        #dppl_atomic_add3 = dppl.kernel('void(uint32[:,:])')(atomic_add3)
        dppl_atomic_add3 = dppl.kernel(atomic_add3)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            dppl_atomic_add3[(4, 8), dppl.DEFAULT_LOCAL_SIZE](ary)

        self.assertTrue(np.all(ary == orig + 1))


if __name__ == '__main__':
    unittest.main()
