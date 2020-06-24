from __future__ import print_function, division, absolute_import

import numpy as np

from numba import dppy
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase
import dppy.core as ocldrv


@unittest.skipUnless(ocldrv.has_gpu_device, 'test only on GPU system')
class TestDPPYFunc(DPPYTestCase):
    N = 256

    def test_dppy_func_device_array(self):
        @dppy.func
        def g(a):
            return a + 1

        @dppy.kernel
        def f(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i])

        a = np.ones(self.N)
        b = np.ones(self.N)

        with ocldrv.igpu_context(0) as device_env:
            # Copy data to the device
            dA = device_env.copy_array_to_device(a)
            dB = device_env.copy_array_to_device(b)

            f[self.N, dppy.DEFAULT_LOCAL_SIZE](dA, dB)

            # Copy data from the device
            device_env.copy_array_from_device(dB)

        self.assertTrue(np.all(b == 2))

    def test_dppy_func_ndarray(self):
        @dppy.func
        def g(a):
            return a + 1

        @dppy.kernel
        def f(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i])

        @dppy.kernel
        def h(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i]) + 1

        a = np.ones(self.N)
        b = np.ones(self.N)

        with ocldrv.igpu_context(0) as device_env:
            f[self.N, dppy.DEFAULT_LOCAL_SIZE](a, b)

            self.assertTrue(np.all(b == 2))

            h[self.N, dppy.DEFAULT_LOCAL_SIZE](a, b)

            self.assertTrue(np.all(b == 3))


if __name__ == '__main__':
    unittest.main()
