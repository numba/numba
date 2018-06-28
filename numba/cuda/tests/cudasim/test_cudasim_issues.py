from __future__ import absolute_import, print_function, division

import threading

import numpy as np

from numba import unittest_support as unittest
from numba import cuda
from numba.cuda.testing import SerialMixin
import numba.cuda.simulator as simulator


class TestCudaSimIssues(SerialMixin, unittest.TestCase):

    def test_cuda_module_in_device_function(self):
        """
        Discovered in https://github.com/numba/numba/issues/1837.
        When the `cuda` module is referenced in a device function,
        it does not have the kernel API (e.g. cuda.threadIdx, cuda.shared)
        """
        from .support import cuda_module_in_device_function as inner

        @cuda.jit
        def outer(out):
            tid = inner()
            if tid < out.size:
                out[tid] = tid

        arr = np.zeros(10, dtype=np.int32)
        outer[1, 11](arr)
        expected = np.arange(arr.size, dtype=np.int32)
        np.testing.assert_equal(expected, arr)

    def test_deadlock_on_exception(self):
        def assert_no_blockthreads():
            blockthreads = [
                t for t in threading.enumerate()
                if isinstance(t, simulator.kernel.BlockThread)
            ]

            self.assertListEqual(blockthreads, [])

        @simulator.jit
        def assign_with_sync(x, y):
            i = cuda.grid(1)
            y[i] = x[i]

            cuda.syncthreads()

        x = np.arange(20)
        y = np.empty(20)
        assign_with_sync[1, 20](x, y)
        np.testing.assert_array_equal(x, y)
        assert_no_blockthreads()


        with self.assertRaises(IndexError):
            assign_with_sync[1, 30](x, y)
        assert_no_blockthreads()


if __name__ == '__main__':
    unittest.main()
