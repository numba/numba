from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy
import dppy.core as ocldrv
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase


def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


class TestCaching(DPPYTestCase):
    def test_caching_kernel(self):
        global_size = 10
        N = global_size

        a = np.array(np.random.random(N), dtype=np.float32)
        b = np.array(np.random.random(N), dtype=np.float32)
        c = np.ones_like(a)

        device_env = None

        try:
            device_env = ocldrv.runtime.get_gpu_device()
            print("Selected GPU device")
        except:
            try:
                device_env = ocldrv.runtime.get_cpu_device()
                print("Selected CPU device")
            except:
                print("No OpenCL devices found on the system")
                raise SystemExit()

        # Copy the data to the device
        dA = device_env.copy_array_to_device(a)
        dB = device_env.copy_array_to_device(b)
        dC = ocldrv.DeviceArray(device_env.get_env_ptr(), c)

        func = dppy.kernel(data_parallel_sum)
        caching_kernel = func[device_env, global_size].specialize(dA, dB, dC)

        for i in range(10):
            cached_kernel = func[device_env, global_size].specialize(dA, dB, dC)
            self.assertIs(caching_kernel, cached_kernel)


if __name__ == '__main__':
    unittest.main()
