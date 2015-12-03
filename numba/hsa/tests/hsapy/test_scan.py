from __future__ import print_function, absolute_import, division

import numpy as np

from numba import unittest_support as unittest
from numba import hsa, intp


@hsa.jit(device=True)
def device_scan_generic(tid, data):
    """Inclusive prefix sum within a single block

    Requires tid should have range [0, data.size) and data.size must be
    power of 2.
    """
    n = data.size

    # Upsweep
    offset = 1
    d = n // 2
    while d > 0:
        hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)
        if tid < d:
            ai = offset * (2 * tid + 1) - 1
            bi = offset * (2 * tid + 2) - 1
            data[bi] += data[ai]

        offset *= 2
        d //= 2

    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)
    prefixsum = data[n - 1]
    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)
    if tid == 0:
        data[n - 1] = 0

    # Downsweep
    d = 1
    offset = n
    while d < n:
        offset //= 2
        hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)
        if tid < d:
            ai = offset * (2 * tid + 1) - 1
            bi = offset * (2 * tid + 2) - 1

            tmp = data[ai]
            data[ai] = data[bi]
            data[bi] += tmp

        d *= 2

    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)
    return prefixsum


_WARPSIZE = 64


@hsa.jit(device=True)
def warp_scan(tid, temp, inclusive):
    """Intra-warp scan

    Note
    ----
    Assume all threads are in lockstep
    """
    hsa.wavebarrier()
    lane = tid & (_WARPSIZE - 1)
    if lane >= 1:
        temp[tid] += temp[tid - 1]

    hsa.wavebarrier()
    if lane >= 2:
        temp[tid] += temp[tid - 2]

    hsa.wavebarrier()
    if lane >= 4:
        temp[tid] += temp[tid - 4]

    hsa.wavebarrier()
    if lane >= 8:
        temp[tid] += temp[tid - 8]

    hsa.wavebarrier()
    if lane >= 16:
        temp[tid] += temp[tid - 16]

    hsa.wavebarrier()
    if lane >= 32:
        temp[tid] += temp[tid - 32]

    hsa.wavebarrier()
    if inclusive:
        return temp[tid]
    else:
        return temp[tid - 1] if lane > 0 else 0


@hsa.jit(device=True)
def device_scan(tid, data, temp, inclusive):
    """
    Args
    ----
    tid:
        thread id
    data: scalar
        input for tid
    temp: shared memory for temporary work
    """
    lane = tid & (_WARPSIZE - 1)
    warpid = tid >> 6

    # Preload
    temp[tid] = data
    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)

    # Scan warps in parallel
    warp_scan_res = warp_scan(tid, temp, inclusive)
    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)

    # Get parital result
    if lane == (_WARPSIZE - 1):
        temp[warpid] = temp[tid]
    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)

    # Scan the partial results
    if warpid == 0:
        warp_scan(tid, temp, True)
    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)

    # Accumlate scanned partial results
    if warpid > 0:
        warp_scan_res += temp[warpid - 1]
    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)

    # Output
    if tid == temp.size - 1:
        # Last thread computes prefix sum
        if inclusive:
            temp[0] = warp_scan_res
        else:
            temp[0] = warp_scan_res + data

    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)

    # Load prefixsum
    prefixsum = temp[0]
    hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)

    return warp_scan_res, prefixsum


@hsa.jit(device=True)
def shuffle_up(val, width):
    tid = hsa.get_local_id(0)
    hsa.wavebarrier()
    res = hsa.activelanepermute_wavewidth(val, tid - width, 0, False)
    return res


@hsa.jit(device=True)
def shuf_wave_inclusive_scan(val):
    tid = hsa.get_local_id(0)
    lane = tid & (_WARPSIZE - 1)

    hsa.wavebarrier()
    shuf = shuffle_up(val, 1)
    if lane >= 1:
        val += shuf

    hsa.wavebarrier()
    shuf = shuffle_up(val, 2)
    if lane >= 2:
        val += shuf

    hsa.wavebarrier()
    shuf = shuffle_up(val, 4)
    if lane >= 4:
        val += shuf

    hsa.wavebarrier()
    shuf = shuffle_up(val, 8)
    if lane >= 8:
        val += shuf

    hsa.wavebarrier()
    shuf = shuffle_up(val, 16)
    if lane >= 16:
        val += shuf

    hsa.wavebarrier()
    shuf = shuffle_up(val, 32)
    if lane >= 32:
        val += shuf

    hsa.wavebarrier()
    return val


@hsa.jit(device=True)
def shuf_device_inclusive_scan(data, temp):
    """
    Args
    ----
    data: scalar
        input for tid
    temp: shared memory for temporary work, requires at least
    threadcount/wavesize storage
    """
    tid = hsa.get_local_id(0)
    lane = tid & (_WARPSIZE - 1)
    warpid = tid >> 6

    # Scan warps in parallel
    warp_scan_res = shuf_wave_inclusive_scan(data)

    hsa.barrier()

    # Store partial sum into shared memory
    if lane == (_WARPSIZE - 1):
        temp[warpid] = warp_scan_res

    hsa.barrier()

    # Scan the partial sum by first wave
    if warpid == 0:
        shuf_wave_inclusive_scan(temp[lane])

    hsa.barrier()

    # Get block sum for each wave
    blocksum = 0    # first wave is 0
    if warpid > 0:
        blocksum = temp[warpid - 1]

    return warp_scan_res + blocksum


class TestScan(unittest.TestCase):
    def test_single_block(self):

        @hsa.jit
        def scan_block(data, sums):
            sm_data = hsa.shared.array(64, dtype=intp)
            tid = hsa.get_local_id(0)
            gid = hsa.get_global_id(0)
            blkid = hsa.get_group_id(0)
            sm_data[tid] = data[gid]
            prefixsum = device_scan_generic(tid, sm_data)
            data[gid] = sm_data[tid]
            if tid == 0:
                sums[blkid] = prefixsum

        data = np.random.randint(0, 4, size=64).astype(np.intp)
        expected = data.cumsum()
        sums = np.zeros(1, dtype=np.intp)
        scan_block[1, 64](data, sums)
        np.testing.assert_equal(expected[:-1], data[1:])
        self.assertEqual(expected[-1], sums[0])
        self.assertEqual(0, data[0])

    def test_multi_block(self):

        @hsa.jit
        def scan_block(data, sums):
            sm_data = hsa.shared.array(64, dtype=intp)
            tid = hsa.get_local_id(0)
            gid = hsa.get_global_id(0)
            blkid = hsa.get_group_id(0)
            sm_data[tid] = data[gid]
            prefixsum = device_scan_generic(tid, sm_data)
            data[gid] = sm_data[tid]
            if tid == 0:
                sums[blkid] = prefixsum

        nd_data = np.random.randint(0, 4, size=3 * 64).astype(
            np.intp).reshape(3, 64)
        nd_expected = nd_data.cumsum(axis=1)
        sums = np.zeros(3, dtype=np.intp)
        scan_block[3, 64](nd_data.ravel(), sums)

        for nd in range(nd_expected.shape[0]):
            expected = nd_expected[nd]
            data = nd_data[nd]
            np.testing.assert_equal(expected[:-1], data[1:])
            self.assertEqual(expected[-1], sums[nd])
            self.assertEqual(0, data[0])

    def test_multi_large_block(self):
        @hsa.jit
        def scan_block(data, sums):
            sm_data = hsa.shared.array(128, dtype=intp)
            tid = hsa.get_local_id(0)
            gid = hsa.get_global_id(0)
            blkid = hsa.get_group_id(0)
            sm_data[tid] = data[gid]
            prefixsum = device_scan_generic(tid, sm_data)
            data[gid] = sm_data[tid]
            sums[blkid, tid] = prefixsum

        nd_data = np.random.randint(0, 4, size=3 * 128).astype(
            np.intp).reshape(3, 128)
        nd_expected = nd_data.cumsum(axis=1)
        sums = np.zeros((3, 128), dtype=np.intp)
        scan_block[3, 128](nd_data.ravel(), sums)

        for nd in range(nd_expected.shape[0]):
            expected = nd_expected[nd]
            data = nd_data[nd]
            np.testing.assert_equal(expected[:-1], data[1:])
            np.testing.assert_equal(expected[-1], sums[nd])
            self.assertEqual(0, data[0])


class TestFasterScan(unittest.TestCase):
    def test_single_block(self):
        @hsa.jit
        def scan_block(data, sums):
            sm_data = hsa.shared.array(64, dtype=intp)
            tid = hsa.get_local_id(0)
            gid = hsa.get_global_id(0)
            blkid = hsa.get_group_id(0)

            scanval, prefixsum = device_scan(tid, data[gid], sm_data,
                                             False)

            data[gid] = scanval
            if tid == 0:
                sums[blkid] = prefixsum

        data = np.random.randint(0, 4, size=64).astype(np.intp)
        expected = data.cumsum()
        sums = np.zeros(1, dtype=np.intp)
        scan_block[1, 64](data, sums)
        np.testing.assert_equal(expected[:-1], data[1:])
        self.assertEqual(expected[-1], sums[0])
        self.assertEqual(0, data[0])

    def test_single_larger_block(self):
        @hsa.jit
        def scan_block(data, sums):
            sm_data = hsa.shared.array(256, dtype=intp)
            tid = hsa.get_local_id(0)
            gid = hsa.get_global_id(0)
            blkid = hsa.get_group_id(0)

            scanval, prefixsum = device_scan(tid, data[gid], sm_data,
                                             False)
            data[gid] = scanval
            if tid == 0:
                sums[blkid] = prefixsum

        data = np.random.randint(0, 4, size=256).astype(np.intp)
        expected = data.cumsum()
        sums = np.zeros(1, dtype=np.intp)
        scan_block[1, 256](data, sums)
        np.testing.assert_equal(expected[:-1], data[1:])
        print(data)
        print(sums)
        self.assertEqual(expected[-1], sums[0])
        self.assertEqual(0, data[0])

    def test_multi_large_block(self):
        @hsa.jit
        def scan_block(data, sums):
            sm_data = hsa.shared.array(128, dtype=intp)
            tid = hsa.get_local_id(0)
            gid = hsa.get_global_id(0)
            blkid = hsa.get_group_id(0)

            scanval, prefixsum = device_scan(tid, data[gid], sm_data,
                                             False)

            data[gid] = scanval
            sums[blkid, tid] = prefixsum

        nd_data = np.random.randint(0, 4, size=3 * 128).astype(
            np.intp).reshape(3, 128)
        nd_expected = nd_data.cumsum(axis=1)
        sums = np.zeros((3, 128), dtype=np.intp)
        scan_block[3, 128](nd_data.ravel(), sums)

        for nd in range(nd_expected.shape[0]):
            expected = nd_expected[nd]
            data = nd_data[nd]
            np.testing.assert_equal(expected[:-1], data[1:])
            np.testing.assert_equal(expected[-1], sums[nd])
            self.assertEqual(0, data[0])


class TestShuffleScan(unittest.TestCase):
    def test_shuffle(self):
        @hsa.jit
        def foo(inp, mask, out):
            tid = hsa.get_local_id(0)
            out[tid] = hsa.activelanepermute_wavewidth(inp[tid], mask[tid], 0,
                                                       False)

        inp = np.arange(64, dtype=np.intp)
        for i in range(10):
            mask = np.random.randint(0, inp.size, inp.size).astype(np.uint32)
            out = np.zeros_like(inp)
            foo[1, 64](inp, mask, out)
            np.testing.assert_equal(inp[mask], out)

    def test_shuffle_up(self):
        @hsa.jit
        def foo(inp, out):
            gid = hsa.get_global_id(0)
            out[gid] = shuffle_up(inp[gid], 1)

        inp = np.arange(128, dtype=np.intp)
        out = np.zeros_like(inp)
        foo[1, 128](inp, out)

        inp = inp.reshape(2, 64)
        out = out.reshape(inp.shape)

        for i in range(out.shape[0]):
            np.testing.assert_equal(inp[0, :-1], out[0, 1:])
            np.testing.assert_equal(inp[0, -1], out[0, 0])

    def test_shuf_wave_inclusive_scan(self):
        @hsa.jit
        def foo(inp, out):
            gid = hsa.get_global_id(0)
            out[gid] = shuf_wave_inclusive_scan(inp[gid])

        inp = np.arange(64, dtype=np.intp)
        out = np.zeros_like(inp)
        foo[1, 64](inp, out)
        np.testing.assert_equal(out, inp.cumsum())

    def test_shuf_device_inclusive_scan(self):
        @hsa.jit
        def foo(inp, out):
            gid = hsa.get_global_id(0)
            temp = hsa.shared.array(2, dtype=intp)
            out[gid] = shuf_device_inclusive_scan(inp[gid], temp)

        inp = np.arange(128, dtype=np.intp)
        out = np.zeros_like(inp)

        foo[1, inp.size](inp, out)
        np.testing.assert_equal(out, np.cumsum(inp))

if __name__ == '__main__':
    unittest.main()
