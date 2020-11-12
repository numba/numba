from __future__ import print_function, division, absolute_import

import numpy as np

from numba import dppl
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase
import dpctl


@unittest.skipUnless(dpctl.has_gpu_queues(), 'test only on GPU system')
class TestDPPLFunc(DPPLTestCase):
    N = 257

    def test_dppl_func_device_array(self):
        @dppl.func
        def g(a):
            return a + 1

        @dppl.kernel
        def f(a, b):
            i = dppl.get_global_id(0)
            b[i] = g(a[i])

        a = np.ones(self.N)
        b = np.ones(self.N)

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            f[self.N, dppl.DEFAULT_LOCAL_SIZE](a, b)


        self.assertTrue(np.all(b == 2))

    def test_dppl_func_ndarray(self):
        @dppl.func
        def g(a):
            return a + 1

        @dppl.kernel
        def f(a, b):
            i = dppl.get_global_id(0)
            b[i] = g(a[i])

        @dppl.kernel
        def h(a, b):
            i = dppl.get_global_id(0)
            b[i] = g(a[i]) + 1

        a = np.ones(self.N)
        b = np.ones(self.N)

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            f[self.N, dppl.DEFAULT_LOCAL_SIZE](a, b)

            self.assertTrue(np.all(b == 2))

            h[self.N, dppl.DEFAULT_LOCAL_SIZE](a, b)

            self.assertTrue(np.all(b == 3))


if __name__ == '__main__':
    unittest.main()
