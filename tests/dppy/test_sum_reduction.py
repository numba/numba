from __future__ import print_function, division, absolute_import

import numpy as np
import math
import time

from numba import dppy
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase
import dppy.core as ocldrv

@dppy.kernel
def reduction_kernel(A, R, stride):
    i = dppy.get_global_id(0)
    # sum two element
    R[i] = A[i] + A[i+stride]
    # store the sum to be used in nex iteration
    A[i] = R[i]


@unittest.skipUnless(ocldrv.has_gpu_device, 'test only on GPU system')
class TestDPPYSumReduction(DPPYTestCase):
    def test_sum_reduction(self):
        # This test will only work for even case
        N = 1024
        self.assertTrue(N%2 == 0)

        A = np.array(np.random.random(N), dtype=np.float32)
        # at max we will require half the size of A to store sum
        R = np.array(np.random.random(math.ceil(N/2)), dtype=np.float32)

        with ocldrv.igpu_context(0) as device_env:
            # create device array
            dA = device_env.copy_array_to_device(A)
            dR = device_env.copy_array_to_device(R)

            total = N

            while (total > 1):
                # call kernel
                global_size = total // 2
                reduction_kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](dA, dR, global_size)
                total = total // 2

            device_env.copy_array_from_device(dR)
            result = A.sum()
            max_abs_err = result - R[0]
            self.assertTrue(max_abs_err < 1e-4)


if __name__ == '__main__':
    unittest.main()
