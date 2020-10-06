from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppl
import dpctl
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase


def data_parallel_sum(a, b, c):
    i = dppl.get_global_id(0)
    c[i] = a[i] + b[i]


class TestCaching(DPPLTestCase):
    def test_caching_kernel(self):
        global_size = 10
        N = global_size

        a = np.array(np.random.random(N), dtype=np.float32)
        b = np.array(np.random.random(N), dtype=np.float32)
        c = np.ones_like(a)


        with dpctl.device_context("opencl:gpu") as gpu_queue:
            func = dppl.kernel(data_parallel_sum)
            caching_kernel = func[global_size, dppl.DEFAULT_LOCAL_SIZE].specialize(a, b, c)

            for i in range(10):
                cached_kernel = func[global_size, dppl.DEFAULT_LOCAL_SIZE].specialize(a, b, c)
                self.assertIs(caching_kernel, cached_kernel)


if __name__ == '__main__':
    unittest.main()
