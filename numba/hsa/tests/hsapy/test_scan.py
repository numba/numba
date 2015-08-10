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
        hsa.barrier(1)
        if tid < d:
            ai = offset * (2 * tid + 1) - 1
            bi = offset * (2 * tid + 2) - 1
            data[bi] += data[ai]

        offset *= 2
        d //= 2

    hsa.barrier(1)
    prefixsum = data[n - 1]
    hsa.barrier(1)
    if tid == 0:
        data[n - 1] = 0

    # Downsweep
    d = 1
    offset = n
    while d < n:
        offset //= 2
        hsa.barrier(1)
        if tid < d:
            ai = offset * (2 * tid + 1) - 1
            bi = offset * (2 * tid + 2) - 1

            tmp = data[ai]
            data[ai] = data[bi]
            data[bi] += tmp

        d *= 2

    hsa.barrier(1)
    return prefixsum


_WARPSIZE = 64


@hsa.jit(device=True)
def warp_scan(tid, temp, inclusive):
    """Intra-warp scan

    Note
    ----
    Assume all threads are in lockstep
    """
    lane = tid & (_WARPSIZE - 1)
    if lane >= 1:
        temp[tid] += temp[tid - 1]

    if lane >= 2:
        temp[tid] += temp[tid - 2]

    if lane >= 4:
        temp[tid] += temp[tid - 4]

    if lane >= 8:
        temp[tid] += temp[tid - 8]

    if lane >= 16:
        temp[tid] += temp[tid - 16]

    if lane >= 32:
        temp[tid] += temp[tid - 32]

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
    hsa.barrier(1)

    # Scan warps in parallel
    warp_scan_res = warp_scan(tid, temp, inclusive)
    hsa.barrier(1)

    # Get parital result
    if lane == (_WARPSIZE - 1):
        temp[warpid] = temp[tid]
    hsa.barrier(1)

    # Scan the partial results
    if warpid == 0:
        warp_scan(tid, temp, True)
    hsa.barrier(1)

    # Accumlate scanned partial results
    if warpid > 0:
        warp_scan_res += temp[warpid - 1]
    hsa.barrier(1)

    # Output
    if tid == temp.size - 1:
        # Last thread computes prefix sum
        if inclusive:
            temp[0] = warp_scan_res
        else:
            temp[0] = warp_scan_res + data

    hsa.barrier(1)

    # Load prefixsum
    prefixsum = temp[0]
    hsa.barrier(1)

    return warp_scan_res, prefixsum


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


if __name__ == '__main__':
    unittest.main()
