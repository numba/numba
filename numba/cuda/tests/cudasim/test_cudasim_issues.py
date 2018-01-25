from __future__ import absolute_import, print_function, division

import numpy as np

from numba import unittest_support as unittest
from numba import cuda
from numba.cuda.testing import SerialMixin


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


if __name__ == '__main__':
    unittest.main()
