from numba import cuda, int32, float64
from numba import numpy_support as nps

from numba.cuda.testing import unittest, SerialMixin

import numpy as np


recordwith2darray = np.dtype([('i', np.int32),
                              ('j', np.float32, (3, 2))])


class TestSharedMemoryIssue(SerialMixin, unittest.TestCase):
    def test_issue_953_sm_linkage_conflict(self):
        @cuda.jit(device=True)
        def inner():
            inner_arr = cuda.shared.array(1, dtype=int32)

        @cuda.jit
        def outer():
            outer_arr = cuda.shared.array(1, dtype=int32)
            inner()

        outer()

    def _check_shared_array_size(self, shape, expected):
        @cuda.jit
        def s(a):
            arr = cuda.shared.array(shape, dtype=int32)
            a[0] = arr.size

        result = np.zeros(1, dtype=np.int32)
        s(result)
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


class TestSharedMemory(SerialMixin, unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
