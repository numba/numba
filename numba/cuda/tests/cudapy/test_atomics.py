import random
import numpy as np

from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import config


def cc_X_or_above(major, minor):
    if not config.ENABLE_CUDASIM:
        return cuda.current_context().device.compute_capability >= (major, minor)
    else:
        return True


def skip_unless_cc_32(fn):
    return unittest.skipUnless(cc_X_or_above(3, 2), "require cc >= 3.2")(fn)

def skip_unless_cc_50(fn):
    return unittest.skipUnless(cc_X_or_above(5, 0), "require cc >= 5.0")(fn)


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
    bin = int(ary[tid] % 32)
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


def atomic_add_double_global(idx, ary):
    tid = cuda.threadIdx.x
    bin = idx[tid] % 32
    cuda.atomic.add(ary, bin, 1.0)


def atomic_add_double_global_2(ary):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    cuda.atomic.add(ary, (tx, ty), 1)


def atomic_add_double_global_3(ary):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    cuda.atomic.add(ary, (tx, uint64(ty)), 1)


def atomic_add_double(idx, ary):
    tid = cuda.threadIdx.x
    sm = cuda.shared.array(32, float64)
    sm[tid] = 0.0
    cuda.syncthreads()
    bin = idx[tid] % 32
    cuda.atomic.add(sm, bin, 1.0)
    cuda.syncthreads()
    ary[tid] = sm[tid]


def atomic_add_double_2(ary):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    sm = cuda.shared.array((4, 8), float64)
    sm[tx, ty] = ary[tx, ty]
    cuda.syncthreads()
    cuda.atomic.add(sm, (tx, ty), 1)
    cuda.syncthreads()
    ary[tx, ty] = sm[tx, ty]


def atomic_add_double_3(ary):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    sm = cuda.shared.array((4, 8), float64)
    sm[tx, ty] = ary[tx, ty]
    cuda.syncthreads()
    cuda.atomic.add(sm, (tx, uint64(ty)), 1)
    cuda.syncthreads()
    ary[tx, ty] = sm[tx, ty]


def atomic_max(res, ary):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    cuda.atomic.max(res, 0, ary[tx, bx])


def atomic_min(res, ary):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    cuda.atomic.min(res, 0, ary[tx, bx])


def atomic_max_double_normalizedindex(res, ary):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    cuda.atomic.max(res, 0, ary[tx, uint64(bx)])


def atomic_max_double_oneindex(res, ary):
    tx = cuda.threadIdx.x
    cuda.atomic.max(res, 0, ary[tx])


def atomic_max_double_shared(res, ary):
    tid = cuda.threadIdx.x
    smary = cuda.shared.array(32, float64)
    smary[tid] = ary[tid]
    smres = cuda.shared.array(1, float64)
    if tid == 0:
        smres[0] = res[0]
    cuda.syncthreads()
    cuda.atomic.max(smres, 0, smary[tid])
    cuda.syncthreads()
    if tid == 0:
        res[0] = smres[0]


def atomic_compare_and_swap(res, old, ary):
    gid = cuda.grid(1)
    if gid < res.size:
        out = cuda.atomic.compare_and_swap(res[gid:], -99, ary[gid])
        old[gid] = out


class TestCudaAtomics(CUDATestCase):
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
        orig = ary.copy().astype(np.intp)
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

    def assertCorrectFloat64Atomics(self, kernel, shared=True):
        if config.ENABLE_CUDASIM:
            return

        asm = kernel.inspect_asm()
        if cc_X_or_above(6, 0):
            if shared:
                self.assertIn('atom.shared.add.f64', asm)
            else:
                self.assertIn('atom.add.f64', asm)
        else:
            if shared:
                self.assertIn('atom.shared.cas.b64', asm)
            else:
                self.assertIn('atom.cas.b64', asm)

    @skip_unless_cc_50
    def test_atomic_add_double(self):
        idx = np.random.randint(0, 32, size=32)
        ary = np.zeros(32, np.float64)
        cuda_func = cuda.jit('void(int64[:], float64[:])')(atomic_add_double)
        cuda_func[1, 32](idx, ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(idx.size):
            gold[idx[i]] += 1.0

        np.testing.assert_equal(ary, gold)
        self.assertCorrectFloat64Atomics(cuda_func)

    def test_atomic_add_double_2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float64).reshape(4, 8)
        orig = ary.copy()
        cuda_func = cuda.jit('void(float64[:,:])')(atomic_add_double_2)
        cuda_func[1, (4, 8)](ary)
        np.testing.assert_equal(ary, orig + 1)
        self.assertCorrectFloat64Atomics(cuda_func)

    def test_atomic_add_double_3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float64).reshape(4, 8)
        orig = ary.copy()
        cuda_func = cuda.jit('void(float64[:,:])')(atomic_add_double_3)
        cuda_func[1, (4, 8)](ary)

        np.testing.assert_equal(ary, orig + 1)
        self.assertCorrectFloat64Atomics(cuda_func)

    @skip_unless_cc_50
    def test_atomic_add_double_global(self):
        idx = np.random.randint(0, 32, size=32)
        ary = np.zeros(32, np.float64)
        cuda_func = cuda.jit('void(int64[:], float64[:])')(atomic_add_double_global)
        cuda_func[1, 32](idx, ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(idx.size):
            gold[idx[i]] += 1.0

        np.testing.assert_equal(ary, gold)
        self.assertCorrectFloat64Atomics(cuda_func, shared=False)

    def test_atomic_add_double_global_2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float64).reshape(4, 8)
        orig = ary.copy()
        cuda_func = cuda.jit('void(float64[:,:])')(atomic_add_double_global_2)
        cuda_func[1, (4, 8)](ary)
        np.testing.assert_equal(ary, orig + 1)
        self.assertCorrectFloat64Atomics(cuda_func, shared=False)

    def test_atomic_add_double_global_3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float64).reshape(4, 8)
        orig = ary.copy()
        cuda_func = cuda.jit('void(float64[:,:])')(atomic_add_double_global_3)
        cuda_func[1, (4, 8)](ary)

        np.testing.assert_equal(ary, orig + 1)
        self.assertCorrectFloat64Atomics(cuda_func, shared=False)

    def check_atomic_max(self, dtype, lo, hi):
        vals = np.random.randint(lo, hi, size=(32, 32)).astype(dtype)
        res = np.zeros(1, dtype=vals.dtype)
        cuda_func = cuda.jit(atomic_max)
        cuda_func[32, 32](res, vals)
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
        cuda_func = cuda.jit(atomic_min)
        cuda_func[32, 32](res, vals)

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
        cuda_func = cuda.jit('void(float64[:], float64[:,:])')(
            atomic_max_double_normalizedindex)
        cuda_func[32, 32](res, vals)

        gold = np.max(vals)
        np.testing.assert_equal(res, gold)

    def test_atomic_max_double_oneindex(self):
        vals = np.random.randint(0, 128, size=32).astype(np.float64)
        res = np.zeros(1, np.float64)
        cuda_func = cuda.jit('void(float64[:], float64[:])')(
            atomic_max_double_oneindex)
        cuda_func[1, 32](res, vals)

        gold = np.max(vals)
        np.testing.assert_equal(res, gold)

    # Taken together, _test_atomic_minmax_nan_location and
    # _test_atomic_minmax_nan_val check that NaNs are treated similarly to the
    # way they are in Python / NumPy - that is, {min,max}(a, b) == a if either
    # a or b is a NaN. For the atomics, this means that the max is taken as the
    # value stored in the memory location rather than the value supplied - i.e.
    # for:
    #
    #    cuda.atomic.{min,max}(ary, idx, val)
    #
    # the result will be ary[idx] for either of ary[idx] or val being NaN.

    def _test_atomic_minmax_nan_location(self, func):

        cuda_func = cuda.jit('void(float64[:], float64[:,:])')(func)

        vals = np.random.randint(0, 128, size=(1,1)).astype(np.float64)
        res = np.zeros(1, np.float64) + np.nan
        cuda_func[1, 1](res, vals)
        np.testing.assert_equal(res, [np.nan])

    def _test_atomic_minmax_nan_val(self, func):
        cuda_func = cuda.jit('void(float64[:], float64[:,:])')(func)

        res = np.random.randint(0, 128, size=1).astype(np.float64)
        gold = res.copy()
        vals = np.zeros((1, 1), np.float64) + np.nan
        cuda_func[1, 1](res, vals)

        np.testing.assert_equal(res, gold)

    def test_atomic_min_nan_location(self):
        self._test_atomic_minmax_nan_location(atomic_min)

    def test_atomic_max_nan_location(self):
        self._test_atomic_minmax_nan_location(atomic_max)

    def test_atomic_min_nan_val(self):
        self._test_atomic_minmax_nan_val(atomic_min)

    def test_atomic_max_nan_val(self):
        self._test_atomic_minmax_nan_val(atomic_max)

    def test_atomic_max_double_shared(self):
        vals = np.random.randint(0, 32, size=32).astype(np.float64)
        res = np.zeros(1, np.float64)
        cuda_func = cuda.jit('void(float64[:], float64[:])')(atomic_max_double_shared)
        cuda_func[1, 32](res, vals)

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

        cuda_func = cuda.jit(atomic_compare_and_swap)
        cuda_func[10, 10](res, out, ary)

        np.testing.assert_array_equal(expect_res, res)
        np.testing.assert_array_equal(expect_out, out)

    # Tests that the atomic add, min, and max operations return the old value -
    # in the simulator, they did not (see Issue #5458). The max and min have
    # special handling for NaN values, so we explicitly test with a NaN in the
    # array being modified and the value provided.

    def _test_atomic_returns_old(self, kernel, initial):
        x = np.zeros(2, dtype=np.float32)
        x[0] = initial
        kernel[1, 1](x)
        if np.isnan(initial):
            self.assertTrue(np.isnan(x[1]))
        else:
            self.assertEqual(x[1], initial)

    def test_atomic_add_returns_old(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.add(x, 0, 1)

        self._test_atomic_returns_old(kernel, 10)

    def test_atomic_max_returns_old(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.max(x, 0, 1)

        self._test_atomic_returns_old(kernel, 10)

    def test_atomic_max_returns_old_nan_in_array(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.max(x, 0, 1)

        self._test_atomic_returns_old(kernel, np.nan)

    def test_atomic_max_returns_old_nan_val(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.max(x, 0, np.nan)

        self._test_atomic_returns_old(kernel, 10)

    def test_atomic_min_returns_old(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.min(x, 0, 11)

        self._test_atomic_returns_old(kernel, 10)

    def test_atomic_min_returns_old_nan_in_array(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.min(x, 0, 11)

        self._test_atomic_returns_old(kernel, np.nan)

    def test_atomic_min_returns_old_nan_val(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.min(x, 0, 11)

        self._test_atomic_returns_old(kernel, np.nan)


if __name__ == '__main__':
    unittest.main()
