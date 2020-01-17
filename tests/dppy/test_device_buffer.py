from __future__ import print_function, division, absolute_import

import numpy as np

from numba import dppy
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase
from numba.dppy.dppy_driver import driver as ocldrv


@dppy.kernel(access_types={"read_write": ["A"]})
def test_device_buff_for_array(A):
    i = dppy.get_global_id(0)
    A[i] = 2.0

@dppy.kernel(access_types={"read_write": ["a"]})
def test_device_buff_for_ctypes(a):
    a = 3


def call_device_buff_for_array_kernel(func, only_gpu, cpu_device_env, gpu_device_env, global_size, local_size, A):
    if not only_gpu:
        if cpu_device_env:
            func[cpu_device_env, global_size, local_size](A)
        else:
            assert(False, "Could not find CPU device")

    if gpu_device_env:
        func[gpu_device_env, global_size, local_size](A)
    else:
        assert(False, "Could not find GPU device")


class TestDPPYDeviceBuffer(DPPYTestCase):
    only_gpu = True
    cpu_device_env = None
    gpu_device_env = None

    global_size = 5, 1
    local_size = 1, 1, 1
    N = global_size[0] * local_size[0]

    try:
        cpu_device_env = ocldrv.runtime.get_cpu_device()
    except:
        print("CPU device not found")
    try:
        gpu_device_env = ocldrv.runtime.get_gpu_device()
    except:
        print("GPU device not found")

    def test_device_buffer(self):
        # testing ndarray
        A = np.array(np.random.random(self.N), dtype=np.float32)

        Validator = np.array(np.random.random(self.N), dtype=np.float32)

        for idx, items in enumerate(Validator):
            Validator[idx] = 2.0

        call_device_buff_for_array_kernel(test_device_buff_for_array, \
            self.only_gpu, \
            self.cpu_device_env, self.gpu_device_env, \
            self.global_size, self.local_size, A)

        self.assertTrue(np.all(A == Validator))

        a = np.int32(2)
        global_size = 1, 1
        call_device_buff_for_array_kernel(test_device_buff_for_ctypes, \
            self.only_gpu, \
            self.cpu_device_env, self.gpu_device_env, \
            global_size, self.local_size, a)

        print("-----", a, "----")
        self.assertTrue(a == 3)



if __name__ == '__main__':
    unittest.main()
