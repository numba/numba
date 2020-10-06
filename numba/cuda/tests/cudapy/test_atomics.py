import random
import numpy as np
from textwrap import dedent

from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import config


def cc_X_or_above(major, minor):
    if not config.ENABLE_CUDASIM:
        ctx = cuda.current_context()
        return ctx.device.compute_capability >= (major, minor)
    else:
        return True


def skip_unless_cc_32(fn):
    return unittest.skipUnless(cc_X_or_above(3, 2), "require cc >= 3.2")(fn)


def skip_unless_cc_50(fn):
    return unittest.skipUnless(cc_X_or_above(5, 0), "require cc >= 5.0")(fn)


@cuda.jit(device=True)
def atomic_cast_to_uint64(num):
    return uint64(num)


@cuda.jit(device=True)
def atomic_cast_to_int(num):
    return int(num)


@cuda.jit(device=True)
def atomic_cast_none(num):
    return num


@cuda.jit(device=True)
def atomic_binary_1dim_shared(ary, idx, op2, ary_dtype, ary_nelements,
                              binop_func, cast_func, initializer):
    tid = cuda.threadIdx.x
    sm = cuda.shared.array(ary_nelements, ary_dtype)
    sm[tid] = initializer
    cuda.syncthreads()
    bin = cast_func(idx[tid] % ary_nelements)
    binop_func(sm, bin, op2)
    cuda.syncthreads()
    ary[tid] = sm[tid]


@cuda.jit(device=True)
def atomic_binary_2dim_shared(ary, op2, ary_dtype, ary_shape,
                              binop_func, y_cast_func):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    sm = cuda.shared.array(ary_shape, ary_dtype)
    sm[tx, ty] = ary[tx, ty]
    cuda.syncthreads()
    binop_func(sm, (tx, y_cast_func(ty)), op2)
    cuda.syncthreads()
    ary[tx, ty] = sm[tx, ty]


@cuda.jit(device=True)
def atomic_binary_2dim_global(ary, op2, binop_func, y_cast_func):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    binop_func(ary, (tx, y_cast_func(ty)), op2)


@cuda.jit(device=True)
def atomic_binary_1dim_global(ary, idx, ary_nelements, op2, binop_func):
    tid = cuda.threadIdx.x
    bin = idx[tid] % ary_nelements
    binop_func(ary, bin, op2)


def atomic_add(ary):
    atomic_binary_1dim_shared(ary, ary, 1, uint32, 32,
                              cuda.atomic.add, atomic_cast_none, 0)


def atomic_add2(ary):
    atomic_binary_2dim_shared(ary, 1, uint32, (4, 8),
                              cuda.atomic.add, atomic_cast_none)


def atomic_add3(ary):
    atomic_binary_2dim_shared(ary, 1, uint32, (4, 8),
                              cuda.atomic.add, atomic_cast_to_uint64)


def atomic_add_float(ary):
    atomic_binary_1dim_shared(ary, ary, 1.0, float32, 32,
                              cuda.atomic.add, atomic_cast_to_int, 0.0)


def atomic_add_float_2(ary):
    atomic_binary_2dim_shared(ary, 1.0, float32, (4, 8),
                              cuda.atomic.add, atomic_cast_none)


def atomic_add_float_3(ary):
    atomic_binary_2dim_shared(ary, 1.0, float32, (4, 8),
                              cuda.atomic.add, atomic_cast_to_uint64)


def atomic_add_double_global(idx, ary):
    atomic_binary_1dim_global(ary, idx, 32, 1.0, cuda.atomic.add)


def atomic_add_double_global_2(ary):
    atomic_binary_2dim_global(ary, 1, cuda.atomic.add, atomic_cast_none)


def atomic_add_double_global_3(ary):
    atomic_binary_2dim_global(ary, 1, cuda.atomic.add, atomic_cast_to_uint64)


def atomic_add_double(idx, ary):
    atomic_binary_1dim_shared(ary, idx, 1.0, float64, 32,
                              cuda.atomic.add, atomic_cast_none, 0.0)


def atomic_add_double_2(ary):
    atomic_binary_2dim_shared(ary, 1.0, float64, (4, 8),
                              cuda.atomic.add, atomic_cast_none)


def atomic_add_double_3(ary):
    atomic_binary_2dim_shared(ary, 1.0, float64, (4, 8),
                              cuda.atomic.add, atomic_cast_to_uint64)


def atomic_sub(ary):
    atomic_binary_1dim_shared(ary, ary, 1, uint32, 32,
                              cuda.atomic.sub, atomic_cast_none, 0)


def atomic_sub2(ary):
    atomic_binary_2dim_shared(ary, 1, uint32, (4, 8),
                              cuda.atomic.sub, atomic_cast_none)


def atomic_sub3(ary):
    atomic_binary_2dim_shared(ary, 1, uint32, (4, 8),
                              cuda.atomic.sub, atomic_cast_to_uint64)


def atomic_sub_float(ary):
    atomic_binary_1dim_shared(ary, ary, 1.0, float32, 32,
                              cuda.atomic.sub, atomic_cast_to_int, 0.0)


def atomic_sub_float_2(ary):
    atomic_binary_2dim_shared(ary, 1.0, float32, (4, 8),
                              cuda.atomic.sub, atomic_cast_none)


def atomic_sub_float_3(ary):
    atomic_binary_2dim_shared(ary, 1.0, float32, (4, 8),
                              cuda.atomic.sub, atomic_cast_to_uint64)


def atomic_sub_double(idx, ary):
    atomic_binary_1dim_shared(ary, idx, 1.0, float64, 32,
                              cuda.atomic.sub, atomic_cast_none, 0.0)


def atomic_sub_double_2(ary):
    atomic_binary_2dim_shared(ary, 1.0, float64, (4, 8),
                              cuda.atomic.sub, atomic_cast_none)


def atomic_sub_double_3(ary):
    atomic_binary_2dim_shared(ary, 1.0, float64, (4, 8),
                              cuda.atomic.sub, atomic_cast_to_uint64)


def atomic_sub_double_global(idx, ary):
    atomic_binary_1dim_global(ary, idx, 32, 1.0, cuda.atomic.sub)


def atomic_sub_double_global_2(ary):
    atomic_binary_2dim_global(ary, 1.0, cuda.atomic.sub, atomic_cast_none)


def atomic_sub_double_global_3(ary):
    atomic_binary_2dim_shared(ary, 1.0, float64, (4, 8),
                              cuda.atomic.sub, atomic_cast_to_uint64)


def gen_atomic_extreme_funcs(func):

    fns = dedent("""
    def atomic(res, ary):
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        {func}(res, 0, ary[tx, bx])

    def atomic_double_normalizedindex(res, ary):
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        {func}(res, 0, ary[tx, uint64(bx)])

    def atomic_double_oneindex(res, ary):
        tx = cuda.threadIdx.x
        {func}(res, 0, ary[tx])

    def atomic_double_shared(res, ary):
        tid = cuda.threadIdx.x
        smary = cuda.shared.array(32, float64)
        smary[tid] = ary[tid]
        smres = cuda.shared.array(1, float64)
        if tid == 0:
            smres[0] = res[0]
        cuda.syncthreads()
        {func}(smres, 0, smary[tid])
        cuda.syncthreads()
        if tid == 0:
            res[0] = smres[0]
    """).format(func=func)
    ld = {}
    exec(fns, {'cuda': cuda, 'float64': float64, 'uint64': uint64}, ld)
    return (ld['atomic'], ld['atomic_double_normalizedindex'],
            ld['atomic_double_oneindex'], ld['atomic_double_shared'])


(atomic_max, atomic_max_double_normalizedindex, atomic_max_double_oneindex,
 atomic_max_double_shared) = gen_atomic_extreme_funcs('cuda.atomic.max')
(atomic_min, atomic_min_double_normalizedindex, atomic_min_double_oneindex,
 atomic_min_double_shared) = gen_atomic_extreme_funcs('cuda.atomic.min')
(atomic_nanmax, atomic_nanmax_double_normalizedindex,
 atomic_nanmax_double_oneindex, atomic_nanmax_double_shared) = \
    gen_atomic_extreme_funcs('cuda.atomic.nanmax')
(atomic_nanmin, atomic_nanmin_double_normalizedindex,
 atomic_nanmin_double_oneindex, atomic_nanmin_double_shared) = \
    gen_atomic_extreme_funcs('cuda.atomic.nanmin')


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
        idx = np.random.randint(0, 32, size=32, dtype=np.int64)
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
        idx = np.random.randint(0, 32, size=32, dtype=np.int64)
        ary = np.zeros(32, np.float64)
        sig = 'void(int64[:], float64[:])'
        cuda_func = cuda.jit(sig)(atomic_add_double_global)
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

    def test_atomic_sub(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32)
        orig = ary.copy()
        cuda_atomic_sub = cuda.jit('void(uint32[:])')(atomic_sub)
        cuda_atomic_sub[1, 32](ary)

        gold = np.zeros(32, dtype=np.uint32)
        for i in range(orig.size):
            gold[orig[i]] -= 1

        self.assertTrue(np.all(ary == gold))

    def test_atomic_sub2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        cuda_atomic_sub2 = cuda.jit('void(uint32[:,:])')(atomic_sub2)
        cuda_atomic_sub2[1, (4, 8)](ary)
        self.assertTrue(np.all(ary == orig - 1))

    def test_atomic_sub3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
        orig = ary.copy()
        cuda_atomic_sub3 = cuda.jit('void(uint32[:,:])')(atomic_sub3)
        cuda_atomic_sub3[1, (4, 8)](ary)
        self.assertTrue(np.all(ary == orig - 1))

    def test_atomic_sub_float(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float32)
        orig = ary.copy().astype(np.intp)
        cuda_atomic_sub_float = cuda.jit('void(float32[:])')(atomic_sub_float)
        cuda_atomic_sub_float[1, 32](ary)

        gold = np.zeros(32, dtype=np.float32)
        for i in range(orig.size):
            gold[orig[i]] -= 1.0

        self.assertTrue(np.all(ary == gold))

    def test_atomic_sub_float_2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float32).reshape(4, 8)
        orig = ary.copy()
        cuda_atomic_sub2 = cuda.jit('void(float32[:,:])')(atomic_sub_float_2)
        cuda_atomic_sub2[1, (4, 8)](ary)
        self.assertTrue(np.all(ary == orig - 1))

    def test_atomic_sub_float_3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float32).reshape(4, 8)
        orig = ary.copy()
        cuda_atomic_sub3 = cuda.jit('void(float32[:,:])')(atomic_sub_float_3)
        cuda_atomic_sub3[1, (4, 8)](ary)
        self.assertTrue(np.all(ary == orig - 1))

    def test_atomic_sub_double(self):
        idx = np.random.randint(0, 32, size=32, dtype=np.int64)
        ary = np.zeros(32, np.float64)
        cuda_func = cuda.jit('void(int64[:], float64[:])')(atomic_sub_double)
        cuda_func[1, 32](idx, ary)

        gold = np.zeros(32, dtype=np.float64)
        for i in range(idx.size):
            gold[idx[i]] -= 1.0

        np.testing.assert_equal(ary, gold)

    def test_atomic_sub_double_2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float64).reshape(4, 8)
        orig = ary.copy()
        cuda_func = cuda.jit('void(float64[:,:])')(atomic_sub_double_2)
        cuda_func[1, (4, 8)](ary)
        np.testing.assert_equal(ary, orig - 1)

    def test_atomic_sub_double_3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float64).reshape(4, 8)
        orig = ary.copy()
        cuda_func = cuda.jit('void(float64[:,:])')(atomic_sub_double_3)
        cuda_func[1, (4, 8)](ary)
        np.testing.assert_equal(ary, orig - 1)

    def test_atomic_sub_double_global(self):
        idx = np.random.randint(0, 32, size=32, dtype=np.int64)
        ary = np.zeros(32, np.float64)
        sig = 'void(int64[:], float64[:])'
        cuda_func = cuda.jit(sig)(atomic_sub_double_global)
        cuda_func[1, 32](idx, ary)

        gold = np.zeros(32, dtype=np.float64)
        for i in range(idx.size):
            gold[idx[i]] -= 1.0

        np.testing.assert_equal(ary, gold)

    def test_atomic_sub_double_global_2(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float64).reshape(4, 8)
        orig = ary.copy()
        cuda_func = cuda.jit('void(float64[:,:])')(atomic_sub_double_global_2)
        cuda_func[1, (4, 8)](ary)
        np.testing.assert_equal(ary, orig - 1)

    def test_atomic_sub_double_global_3(self):
        ary = np.random.randint(0, 32, size=32).astype(np.float64).reshape(4, 8)
        orig = ary.copy()
        cuda_func = cuda.jit('void(float64[:,:])')(atomic_sub_double_global_3)
        cuda_func[1, (4, 8)](ary)
        np.testing.assert_equal(ary, orig - 1)

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

    def test_atomic_min_double_normalizedindex(self):
        vals = np.random.randint(0, 65535, size=(32, 32)).astype(np.float64)
        res = np.ones(1, np.float64) * 65535
        cuda_func = cuda.jit('void(float64[:], float64[:,:])')(
            atomic_min_double_normalizedindex)
        cuda_func[32, 32](res, vals)

        gold = np.min(vals)
        np.testing.assert_equal(res, gold)

    def test_atomic_min_double_oneindex(self):
        vals = np.random.randint(0, 128, size=32).astype(np.float64)
        res = np.ones(1, np.float64) * 128
        cuda_func = cuda.jit('void(float64[:], float64[:])')(
            atomic_min_double_oneindex)
        cuda_func[1, 32](res, vals)

        gold = np.min(vals)
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
        sig = 'void(float64[:], float64[:])'
        cuda_func = cuda.jit(sig)(atomic_max_double_shared)
        cuda_func[1, 32](res, vals)

        gold = np.max(vals)
        np.testing.assert_equal(res, gold)

    def test_atomic_min_double_shared(self):
        vals = np.random.randint(0, 32, size=32).astype(np.float64)
        res = np.ones(1, np.float64) * 32
        sig = 'void(float64[:], float64[:])'
        cuda_func = cuda.jit(sig)(atomic_min_double_shared)
        cuda_func[1, 32](res, vals)

        gold = np.min(vals)
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

    def test_atomic_max_returns_no_replace(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.max(x, 0, 1)

        self._test_atomic_returns_old(kernel, 10)

    def test_atomic_max_returns_old_replace(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.max(x, 0, 10)

        self._test_atomic_returns_old(kernel, 1)

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

    def test_atomic_min_returns_old_no_replace(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.min(x, 0, 11)

        self._test_atomic_returns_old(kernel, 10)

    def test_atomic_min_returns_old_replace(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.min(x, 0, 10)

        self._test_atomic_returns_old(kernel, 11)

    def test_atomic_min_returns_old_nan_in_array(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.min(x, 0, 11)

        self._test_atomic_returns_old(kernel, np.nan)

    def test_atomic_min_returns_old_nan_val(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.min(x, 0, np.nan)

        self._test_atomic_returns_old(kernel, 11)

    # Tests for atomic nanmin/nanmax

    # nanmax tests
    def check_atomic_nanmax(self, dtype, lo, hi):
        vals = np.random.randint(lo, hi, size=(32, 32)).astype(dtype)
        vals[1::2] = np.nan
        res = np.zeros(1, dtype=vals.dtype)
        cuda_func = cuda.jit(atomic_nanmax)
        cuda_func[32, 32](res, vals)
        gold = np.nanmax(vals)
        np.testing.assert_equal(res, gold)

    def test_atomic_nanmax_int32(self):
        self.check_atomic_nanmax(dtype=np.int32, lo=-65535, hi=65535)

    def test_atomic_nanmax_uint32(self):
        self.check_atomic_nanmax(dtype=np.uint32, lo=0, hi=65535)

    @skip_unless_cc_32
    def test_atomic_nanmax_int64(self):
        self.check_atomic_nanmax(dtype=np.int64, lo=-65535, hi=65535)

    @skip_unless_cc_32
    def test_atomic_nanmax_uint64(self):
        self.check_atomic_nanmax(dtype=np.uint64, lo=0, hi=65535)

    def test_atomic_nanmax_float32(self):
        self.check_atomic_nanmax(dtype=np.float32, lo=-65535, hi=65535)

    def test_atomic_nanmax_double(self):
        self.check_atomic_nanmax(dtype=np.float64, lo=-65535, hi=65535)

    def test_atomic_nanmax_double_shared(self):
        vals = np.random.randint(0, 32, size=32).astype(np.float64)
        vals[1::2] = np.nan
        res = np.array([0], dtype=vals.dtype)
        sig = 'void(float64[:], float64[:])'
        cuda_func = cuda.jit(sig)(atomic_nanmax_double_shared)
        cuda_func[1, 32](res, vals)

        gold = np.nanmax(vals)
        np.testing.assert_equal(res, gold)

    def test_atomic_nanmax_double_oneindex(self):
        vals = np.random.randint(0, 128, size=32).astype(np.float64)
        vals[1::2] = np.nan
        res = np.zeros(1, np.float64)
        cuda_func = cuda.jit('void(float64[:], float64[:])')(
            atomic_max_double_oneindex)
        cuda_func[1, 32](res, vals)

        gold = np.nanmax(vals)
        np.testing.assert_equal(res, gold)

    # nanmin tests
    def check_atomic_nanmin(self, dtype, lo, hi):
        vals = np.random.randint(lo, hi, size=(32, 32)).astype(dtype)
        vals[1::2] = np.nan
        res = np.array([65535], dtype=vals.dtype)
        cuda_func = cuda.jit(atomic_nanmin)
        cuda_func[32, 32](res, vals)

        gold = np.nanmin(vals)
        np.testing.assert_equal(res, gold)

    def test_atomic_nanmin_int32(self):
        self.check_atomic_nanmin(dtype=np.int32, lo=-65535, hi=65535)

    def test_atomic_nanmin_uint32(self):
        self.check_atomic_nanmin(dtype=np.uint32, lo=0, hi=65535)

    @skip_unless_cc_32
    def test_atomic_nanmin_int64(self):
        self.check_atomic_nanmin(dtype=np.int64, lo=-65535, hi=65535)

    @skip_unless_cc_32
    def test_atomic_nanmin_uint64(self):
        self.check_atomic_nanmin(dtype=np.uint64, lo=0, hi=65535)

    def test_atomic_nanmin_float(self):
        self.check_atomic_nanmin(dtype=np.float32, lo=-65535, hi=65535)

    def test_atomic_nanmin_double(self):
        self.check_atomic_nanmin(dtype=np.float64, lo=-65535, hi=65535)

    def test_atomic_nanmin_double_shared(self):
        vals = np.random.randint(0, 32, size=32).astype(np.float64)
        vals[1::2] = np.nan
        res = np.array([32], dtype=vals.dtype)
        sig = 'void(float64[:], float64[:])'
        cuda_func = cuda.jit(sig)(atomic_nanmin_double_shared)
        cuda_func[1, 32](res, vals)

        gold = np.nanmin(vals)
        np.testing.assert_equal(res, gold)

    def test_atomic_nanmin_double_oneindex(self):
        vals = np.random.randint(0, 128, size=32).astype(np.float64)
        vals[1::2] = np.nan
        res = np.array([128], np.float64)
        cuda_func = cuda.jit('void(float64[:], float64[:])')(
            atomic_min_double_oneindex)
        cuda_func[1, 32](res, vals)

        gold = np.nanmin(vals)
        np.testing.assert_equal(res, gold)

    # Returning old value tests

    def _test_atomic_nan_returns_old(self, kernel, initial):
        x = np.zeros(2, dtype=np.float32)
        x[0] = initial
        x[1] = np.nan
        kernel[1, 1](x)
        if np.isnan(initial):
            self.assertFalse(np.isnan(x[0]))
            self.assertTrue(np.isnan(x[1]))
        else:
            self.assertEqual(x[1], initial)

    def test_atomic_nanmax_returns_old_no_replace(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.nanmax(x, 0, 1)

        self._test_atomic_nan_returns_old(kernel, 10)

    def test_atomic_nanmax_returns_old_replace(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.nanmax(x, 0, 10)

        self._test_atomic_nan_returns_old(kernel, 1)

    def test_atomic_nanmax_returns_old_nan_in_array(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.nanmax(x, 0, 1)

        self._test_atomic_nan_returns_old(kernel, np.nan)

    def test_atomic_nanmax_returns_old_nan_val(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.nanmax(x, 0, np.nan)

        self._test_atomic_nan_returns_old(kernel, 10)

    def test_atomic_nanmin_returns_old_no_replace(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.nanmin(x, 0, 11)

        self._test_atomic_nan_returns_old(kernel, 10)

    def test_atomic_nanmin_returns_old_replace(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.nanmin(x, 0, 10)

        self._test_atomic_nan_returns_old(kernel, 11)

    def test_atomic_nanmin_returns_old_nan_in_array(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.nanmin(x, 0, 11)

        self._test_atomic_nan_returns_old(kernel, np.nan)

    def test_atomic_nanmin_returns_old_nan_val(self):
        @cuda.jit
        def kernel(x):
            x[1] = cuda.atomic.nanmin(x, 0, np.nan)

        self._test_atomic_nan_returns_old(kernel, 11)


if __name__ == '__main__':
    unittest.main()
