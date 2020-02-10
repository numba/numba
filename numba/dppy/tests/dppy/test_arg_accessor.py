from __future__ import print_function, division, absolute_import

import numpy as np

from numba import dppy
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase
from numba.dppy.dppy_driver import driver as ocldrv

@dppy.kernel(access_types={"read_only": ['a', 'b'], "write_only": ['c'], "read_write": []})
def sum_with_accessor(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]

@dppy.kernel
def sum_without_accessor(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]



def call_kernel(only_gpu, cpu_device_env, gpu_device_env,
        global_size, local_size, A, B, C, func):
    if not only_gpu:
        if cpu_device_env:
            func[cpu_device_env, global_size](A, B, C)
        else:
            assert(False, "Could not find CPU device")

    if gpu_device_env:
        func[gpu_device_env, global_size](A, B, C)
    else:
        assert(False, "Could not find GPU device")



class TestDPPYArgAccessor(DPPYTestCase):
    only_gpu = True
    cpu_device_env = None
    gpu_device_env = None

    global_size = 10
    local_size = 1
    N = global_size * local_size

    try:
        cpu_device_env = ocldrv.runtime.get_cpu_device()
    except:
        print("CPU device not found")
    try:
        gpu_device_env = ocldrv.runtime.get_gpu_device()
    except:
        print("GPU device not found")

    A = np.array(np.random.random(N), dtype=np.float32)
    B = np.array(np.random.random(N), dtype=np.float32)

    def test_arg_with_accessor(self):
        C = np.ones_like(self.A)
        call_kernel(self.only_gpu,
                self.cpu_device_env, self.gpu_device_env,
                self.global_size, self.local_size, self.A,
                self.B, C, sum_with_accessor)
        D = self.A + self.B
        self.assertTrue(np.all(D == C))

    def test_arg_without_accessor(self):
        C = np.ones_like(self.A)
        call_kernel(self.only_gpu,
                self.cpu_device_env, self.gpu_device_env,
                self.global_size, self.local_size, self.A,
                self.B, C, sum_without_accessor)
        D = self.A + self.B
        self.assertTrue(np.all(D == C))

if __name__ == '__main__':
    unittest.main()
