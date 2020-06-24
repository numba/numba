from __future__ import print_function, division, absolute_import

import numpy as np

from numba import dppy
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase
import dppy as ocldrv


@dppy.kernel(access_types={"read_only": ['a', 'b'], "write_only": ['c'], "read_write": []})
def sum_with_accessor(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]

@dppy.kernel
def sum_without_accessor(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]

def call_kernel(global_size, local_size,
                A, B, C, func):
        func[global_size, dppy.DEFAULT_LOCAL_SIZE](A, B, C)


global_size = 10
local_size = 1
N = global_size * local_size

A = np.array(np.random.random(N), dtype=np.float32)
B = np.array(np.random.random(N), dtype=np.float32)
D = A + B


@unittest.skipUnless(ocldrv.has_cpu_device, 'test only on CPU system')
class TestDPPYArgAccessorCPU(DPPYTestCase):
    def test_arg_with_accessor(self):
        C = np.ones_like(A)
        with ocldrv.cpu_context(0) as device_env:
            call_kernel(global_size, local_size,
                        A, B, C, sum_with_accessor)
        self.assertTrue(np.all(D == C))

    def test_arg_without_accessor(self):
        C = np.ones_like(A)
        with ocldrv.cpu_context(0) as device_env:
            call_kernel(global_size, local_size,
                        A, B, C, sum_without_accessor)
        self.assertTrue(np.all(D == C))


@unittest.skipUnless(ocldrv.has_gpu_device, 'test only on GPU system')
class TestDPPYArgAccessorGPU(DPPYTestCase):
    def test_arg_with_accessor(self):
        C = np.ones_like(A)
        with ocldrv.igpu_context(0) as device_env:
            call_kernel(global_size, local_size,
                        A, B, C, sum_with_accessor)
        self.assertTrue(np.all(D == C))

    def test_arg_without_accessor(self):
        C = np.ones_like(A)
        with ocldrv.igpu_context(0) as device_env:
            call_kernel(global_size, local_size,
                        A, B, C, sum_without_accessor)
        self.assertTrue(np.all(D == C))


if __name__ == '__main__':
    unittest.main()
