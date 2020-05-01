from __future__ import print_function, division, absolute_import

import numpy as np

from numba import dppy
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase
import dppy.core as ocldrv

class TestAtomicOp(DPPYTestCase):
    gpu_device_env = None

    try:
        gpu_device_env = ocldrv.runtime.get_gpu_device()
    except:
        print("GPU device not found")

    def test_atomic_add(self):
        @dppy.kernel
        def atomic_add(B):
            i = dppy.get_global_id(0)
            dppy.atomic.add(B, 0, 1)

        N = 100
        B = np.array([0])

        atomic_add[self.gpu_device_env, N](B)

        self.assertTrue(B[0] == N)


    def test_atomic_sub(self):
        @dppy.kernel
        def atomic_sub(B):
            i = dppy.get_global_id(0)
            dppy.atomic.sub(B, 0, 1)

        N = 100
        B = np.array([100])

        atomic_sub[self.gpu_device_env, N](B)

        self.assertTrue(B[0] == 0)



if __name__ == '__main__':
    unittest.main()
