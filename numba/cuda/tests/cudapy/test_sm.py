from numba import cuda, int32, float64

from numba.cuda.testing import unittest, CUDATestCase

import numpy as np
from numba.np import numpy_support as nps


recordwith2darray = np.dtype([('i', np.int32),
                              ('j', np.float32, (3, 2))])


class TestSharedMemoryIssue(CUDATestCase):
    def test_issue_953_sm_linkage_conflict(self):
        @cuda.jit(device=True)
        def inner():
            inner_arr = cuda.shared.array(1, dtype=int32)  # noqa: F841

        @cuda.jit
        def outer():
            outer_arr = cuda.shared.array(1, dtype=int32)  # noqa: F841
            inner()

        outer[1, 1]()

    def _check_shared_array_size(self, shape, expected):
        @cuda.jit
        def s(a):
            arr = cuda.shared.array(shape, dtype=int32)
            a[0] = arr.size

        result = np.zeros(1, dtype=np.int32)
        s[1, 1](result)
        self.assertEqual(result[0], expected)

    def test_issue_1051_shared_size_broken_1d(self):
        self._check_shared_array_size(2, 2)

    def test_issue_1051_shared_size_broken_2d(self):
        self._check_shared_array_size((2, 3), 6)

    def test_issue_1051_shared_size_broken_3d(self):
        self._check_shared_array_size((2, 3, 4), 24)

    def test_issue_2393(self):
        """
        Test issue of warp misalign address due to nvvm not knowing the
        alignment(? but it should have taken the natural alignment of the type)
        """
        num_weights = 2
        num_blocks = 48
        examples_per_block = 4
        threads_per_block = 1

        @cuda.jit
        def costs_func(d_block_costs):
            s_features = cuda.shared.array((examples_per_block, num_weights),
                                           float64)
            s_initialcost = cuda.shared.array(7, float64)  # Bug

            threadIdx = cuda.threadIdx.x

            prediction = 0
            for j in range(num_weights):
                prediction += s_features[threadIdx, j]

            d_block_costs[0] = s_initialcost[0] + prediction

        block_costs = np.zeros(num_blocks, dtype=np.float64)
        d_block_costs = cuda.to_device(block_costs)

        costs_func[num_blocks, threads_per_block](d_block_costs)

        cuda.synchronize()


class TestSharedMemory(CUDATestCase):
    def _test_shared(self, arr):
        # Use a kernel that copies via shared memory to check loading and
        # storing different dtypes with shared memory. All threads in a block
        # collaborate to load in values, then the output values are written
        # only by the first thread in the block after synchronization.

        nelem = len(arr)
        nthreads = 16
        nblocks = int(nelem / nthreads)
        dt = nps.from_dtype(arr.dtype)

        @cuda.jit
        def use_sm_chunk_copy(x, y):
            sm = cuda.shared.array(nthreads, dtype=dt)

            tx = cuda.threadIdx.x
            bx = cuda.blockIdx.x
            bd = cuda.blockDim.x

            # Load this block's chunk into shared
            i = bx * bd + tx
            if i < len(x):
                sm[tx] = x[i]

            cuda.syncthreads()

            # One thread per block writes this block's chunk
            if tx == 0:
                for j in range(nthreads):
                    y[bd * bx + j] = sm[j]

        d_result = cuda.device_array_like(arr)
        use_sm_chunk_copy[nblocks, nthreads](arr, d_result)
        host_result = d_result.copy_to_host()
        np.testing.assert_array_equal(arr, host_result)

    def test_shared_recarray(self):
        arr = np.recarray(128, dtype=recordwith2darray)
        for x in range(len(arr)):
            arr[x].i = x
            j = np.arange(3 * 2, dtype=np.float32)
            arr[x].j = j.reshape(3, 2) * x

        self._test_shared(arr)

    def test_shared_bool(self):
        arr = np.random.randint(2, size=(1024,), dtype=np.bool_)
        self._test_shared(arr)

    def _test_dynshared_slice(self, func, arr, expected):
        # Check that slices of shared memory are correct
        # (See Bug #5073 - prior to the addition of these tests and
        # corresponding fix, slices of dynamic shared arrays all aliased each
        # other)
        nshared = arr.size * arr.dtype.itemsize
        func[1, 1, 0, nshared](arr)
        np.testing.assert_array_equal(expected, arr)

    def test_dynshared_slice_write(self):
        # Test writing values into disjoint slices of dynamic shared memory
        @cuda.jit
        def slice_write(x):
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[0:1]
            sm2 = dynsmem[1:2]

            sm1[0] = 1
            sm2[0] = 2
            x[0] = dynsmem[0]
            x[1] = dynsmem[1]

        arr = np.zeros(2, dtype=np.int32)
        expected = np.array([1, 2], dtype=np.int32)
        self._test_dynshared_slice(slice_write, arr, expected)

    def test_dynshared_slice_read(self):
        # Test reading values from disjoint slices of dynamic shared memory
        @cuda.jit
        def slice_read(x):
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[0:1]
            sm2 = dynsmem[1:2]

            dynsmem[0] = 1
            dynsmem[1] = 2
            x[0] = sm1[0]
            x[1] = sm2[0]

        arr = np.zeros(2, dtype=np.int32)
        expected = np.array([1, 2], dtype=np.int32)
        self._test_dynshared_slice(slice_read, arr, expected)

    def test_dynshared_slice_diff_sizes(self):
        # Test reading values from disjoint slices of dynamic shared memory
        # with different sizes
        @cuda.jit
        def slice_diff_sizes(x):
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[0:1]
            sm2 = dynsmem[1:3]

            dynsmem[0] = 1
            dynsmem[1] = 2
            dynsmem[2] = 3
            x[0] = sm1[0]
            x[1] = sm2[0]
            x[2] = sm2[1]

        arr = np.zeros(3, dtype=np.int32)
        expected = np.array([1, 2, 3], dtype=np.int32)
        self._test_dynshared_slice(slice_diff_sizes, arr, expected)

    def test_dynshared_slice_overlap(self):
        # Test reading values from overlapping slices of dynamic shared memory
        @cuda.jit
        def slice_overlap(x):
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[0:2]
            sm2 = dynsmem[1:4]

            dynsmem[0] = 1
            dynsmem[1] = 2
            dynsmem[2] = 3
            dynsmem[3] = 4
            x[0] = sm1[0]
            x[1] = sm1[1]
            x[2] = sm2[0]
            x[3] = sm2[1]
            x[4] = sm2[2]

        arr = np.zeros(5, dtype=np.int32)
        expected = np.array([1, 2, 2, 3, 4], dtype=np.int32)
        self._test_dynshared_slice(slice_overlap, arr, expected)

    def test_dynshared_slice_gaps(self):
        # Test writing values to slices of dynamic shared memory doesn't write
        # outside the slice
        @cuda.jit
        def slice_gaps(x):
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[1:3]
            sm2 = dynsmem[4:6]

            # Initial values for dynamic shared memory, some to be overwritten
            dynsmem[0] = 99
            dynsmem[1] = 99
            dynsmem[2] = 99
            dynsmem[3] = 99
            dynsmem[4] = 99
            dynsmem[5] = 99
            dynsmem[6] = 99

            sm1[0] = 1
            sm1[1] = 2
            sm2[0] = 3
            sm2[1] = 4

            x[0] = dynsmem[0]
            x[1] = dynsmem[1]
            x[2] = dynsmem[2]
            x[3] = dynsmem[3]
            x[4] = dynsmem[4]
            x[5] = dynsmem[5]
            x[6] = dynsmem[6]

        arr = np.zeros(7, dtype=np.int32)
        expected = np.array([99, 1, 2, 99, 3, 4, 99], dtype=np.int32)
        self._test_dynshared_slice(slice_gaps, arr, expected)

    def test_dynshared_slice_write_backwards(self):
        # Test writing values into disjoint slices of dynamic shared memory
        # with negative steps
        @cuda.jit
        def slice_write_backwards(x):
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[1::-1]
            sm2 = dynsmem[3:1:-1]

            sm1[0] = 1
            sm1[1] = 2
            sm2[0] = 3
            sm2[1] = 4
            x[0] = dynsmem[0]
            x[1] = dynsmem[1]
            x[2] = dynsmem[2]
            x[3] = dynsmem[3]

        arr = np.zeros(4, dtype=np.int32)
        expected = np.array([2, 1, 4, 3], dtype=np.int32)
        self._test_dynshared_slice(slice_write_backwards, arr, expected)

    def test_dynshared_slice_nonunit_stride(self):
        # Test writing values into slice of dynamic shared memory with
        # non-unit stride
        @cuda.jit
        def slice_nonunit_stride(x):
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[::2]

            # Initial values for dynamic shared memory, some to be overwritten
            dynsmem[0] = 99
            dynsmem[1] = 99
            dynsmem[2] = 99
            dynsmem[3] = 99
            dynsmem[4] = 99
            dynsmem[5] = 99

            sm1[0] = 1
            sm1[1] = 2
            sm1[2] = 3

            x[0] = dynsmem[0]
            x[1] = dynsmem[1]
            x[2] = dynsmem[2]
            x[3] = dynsmem[3]
            x[4] = dynsmem[4]
            x[5] = dynsmem[5]

        arr = np.zeros(6, dtype=np.int32)
        expected = np.array([1, 99, 2, 99, 3, 99], dtype=np.int32)
        self._test_dynshared_slice(slice_nonunit_stride, arr, expected)

    def test_dynshared_slice_nonunit_reverse_stride(self):
        # Test writing values into slice of dynamic shared memory with
        # reverse non-unit stride
        @cuda.jit
        def slice_nonunit_reverse_stride(x):
            dynsmem = cuda.shared.array(0, dtype=int32)
            sm1 = dynsmem[-1::-2]

            # Initial values for dynamic shared memory, some to be overwritten
            dynsmem[0] = 99
            dynsmem[1] = 99
            dynsmem[2] = 99
            dynsmem[3] = 99
            dynsmem[4] = 99
            dynsmem[5] = 99

            sm1[0] = 1
            sm1[1] = 2
            sm1[2] = 3

            x[0] = dynsmem[0]
            x[1] = dynsmem[1]
            x[2] = dynsmem[2]
            x[3] = dynsmem[3]
            x[4] = dynsmem[4]
            x[5] = dynsmem[5]

        arr = np.zeros(6, dtype=np.int32)
        expected = np.array([99, 3, 99, 2, 99, 1], dtype=np.int32)
        self._test_dynshared_slice(slice_nonunit_reverse_stride, arr, expected)

    def test_issue_5073(self):
        # An example with which Bug #5073 (slices of dynamic shared memory all
        # alias) was discovered. The kernel uses all threads in the block to
        # load values into slices of dynamic shared memory. One thread per
        # block then writes the loaded values back to a global array after
        # syncthreads().

        arr = np.arange(1024)
        nelem = len(arr)
        nthreads = 16
        nblocks = int(nelem / nthreads)
        dt = nps.from_dtype(arr.dtype)
        nshared = nthreads * arr.dtype.itemsize
        chunksize = int(nthreads / 2)

        @cuda.jit
        def sm_slice_copy(x, y, chunksize):
            dynsmem = cuda.shared.array(0, dtype=dt)
            sm1 = dynsmem[0:chunksize]
            sm2 = dynsmem[chunksize:chunksize * 2]

            tx = cuda.threadIdx.x
            bx = cuda.blockIdx.x
            bd = cuda.blockDim.x

            # load this block's chunk into shared
            i = bx * bd + tx
            if i < len(x):
                if tx < chunksize:
                    sm1[tx] = x[i]
                else:
                    sm2[tx - chunksize] = x[i]

            cuda.syncthreads()

            # one thread per block writes this block's chunk
            if tx == 0:
                for j in range(chunksize):
                    y[bd * bx + j] = sm1[j]
                    y[bd * bx + j + chunksize] = sm2[j]

        d_result = cuda.device_array_like(arr)
        sm_slice_copy[nblocks, nthreads, 0, nshared](arr, d_result, chunksize)
        host_result = d_result.copy_to_host()
        np.testing.assert_array_equal(arr, host_result)


if __name__ == '__main__':
    unittest.main()
