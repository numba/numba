from __future__ import print_function, division, absolute_import

import numpy as np

from numba import dppy
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase
import dppy.core as ocldrv


class TestDPPYFunc(DPPYTestCase):
    N = 10

    device_env = None

    try:
        device_env = ocldrv.runtime.get_gpu_device()
        print("Selected GPU device")
    except:
        print("GPU device not found")
        exit()


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

        # Copy data to the device
        dA = self.device_env.copy_array_to_device(a)
        dB = self.device_env.copy_array_to_device(b)

        f[self.device_env, self.N](dA, dB)

        # Copy data from the device
        self.device_env.copy_array_from_device(dB)

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

        f[self.device_env, self.N](a, b)

        self.assertTrue(np.all(b == 2))

        h[self.device_env, self.N](a, b)

        self.assertTrue(np.all(b == 3))

if __name__ == '__main__':
    unittest.main()
