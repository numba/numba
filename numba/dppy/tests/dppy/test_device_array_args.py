#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy
import dppy as ocldrv
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase

@dppy.kernel
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


global_size = 64
N = global_size

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
d = a + b


@unittest.skipUnless(ocldrv.has_cpu_device, 'test only on CPU system')
class TestDPPYDeviceArrayArgsGPU(DPPYTestCase):
    def test_device_array_args_cpu(self):
        c = np.ones_like(a)

        with ocldrv.cpu_context(0) as device_env:
            # Copy the data to the device
            dA = device_env.copy_array_to_device(a)
            dB = device_env.copy_array_to_device(b)
            dC = device_env.create_device_array(c)
            data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](dA, dB, dC)
            device_env.copy_array_from_device(dC)

            self.assertTrue(np.all(c == d))


@unittest.skipUnless(ocldrv.has_gpu_device, 'test only on GPU system')
class TestDPPYDeviceArrayArgsCPU(DPPYTestCase):
    def test_device_array_args_gpu(self):
        c = np.ones_like(a)

        with ocldrv.igpu_context(0) as device_env:
            # Copy the data to the device
            dA = device_env.copy_array_to_device(a)
            dB = device_env.copy_array_to_device(b)
            dC = device_env.create_device_array(c)
            data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](dA, dB, dC)
            device_env.copy_array_from_device(dC)

        self.assertTrue(np.all(c == d))


if __name__ == '__main__':
    unittest.main()
