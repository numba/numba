#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy
from numba.dppy.dppy_driver import driver as ocldrv
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase

@dppy.kernel
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]

class TestDPPYDeviceArrayArgs(DPPYTestCase):
    global_size = 50, 1
    local_size = 32, 1, 1
    N = global_size[0] * local_size[0]

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    d = a + b

    def test_device_array_args_cpu(self):
        c = np.ones_like(self.a)
        # Select a device for executing the kernel
        device_env = None

        try:
            device_env = ocldrv.runtime.get_cpu_device()
            print("Selected CPU device")
        except:
            print("No OpenCL devices found on the system")
            raise SystemExit()

        # Copy the data to the device
        dA = device_env.copy_array_to_device(self.a)
        dB = device_env.copy_array_to_device(self.b)
        dC = ocldrv.DeviceArray(device_env.get_env_ptr(), c)
        data_parallel_sum[device_env, self.global_size, self.local_size](dA, dB, dC)
        device_env.copy_array_from_device(dC)

        self.assertTrue(np.all(c == self.d))

    def test_device_array_args_gpu(self):
        c = np.ones_like(self.a)

        # Select a device for executing the kernel
        device_env = None

        try:
            device_env = ocldrv.runtime.get_gpu_device()
            print("Selected GPU device")
        except:
            print("No OpenCL devices found on the system")
            raise SystemExit()

        # Copy the data to the device
        dA = device_env.copy_array_to_device(self.a)
        dB = device_env.copy_array_to_device(self.b)
        dC = ocldrv.DeviceArray(device_env.get_env_ptr(), c)
        data_parallel_sum[device_env, self.global_size, self.local_size](dA, dB, dC)
        device_env.copy_array_from_device(dC)

        self.assertTrue(np.all(c == self.d))

if __name__ == '__main__':
    unittest.main()
