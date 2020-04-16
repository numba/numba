from __future__ import print_function, division, absolute_import

import numpy as np

from numba import dppy
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase
import dppy.core as ocldrv

@dppy.kernel
def mul_kernel(A, test):
    i = dppy.get_global_id(0)
    A[i] *= test

def call_mul_device_kernel(only_gpu, cpu_device_env, gpu_device_env, global_size, A, test):
    if not only_gpu:
        if cpu_device_env:
            mul_kernel[cpu_device_env, global_size](A, test)
        else:
            assert(False, "Could not find CPU device")

    if gpu_device_env:
        validator = A * test
        mul_kernel[gpu_device_env, global_size](A, test)
        return validator
    else:
        assert(False, "Could not find GPU device")

class TestDPPYArrayArg(DPPYTestCase):
    only_gpu = True
    cpu_device_env = None
    gpu_device_env = None

    global_size = 10
    N = global_size

    try:
        cpu_device_env = ocldrv.runtime.get_cpu_device()
    except:
        print("CPU device not found")
    try:
        gpu_device_env = ocldrv.runtime.get_gpu_device()
    except:
        print("GPU device not found")

    A = np.array(np.random.random(N), dtype=np.float32)

    def test_integer_arg(self):
        x = np.int32(2)

        validator = call_mul_device_kernel(self.only_gpu, \
                self.cpu_device_env, self.gpu_device_env, \
                self.global_size, self.A, x)
        self.assertTrue(np.all(self.A == validator))

    def test_float_arg(self):
        x = np.float32(2.0)

        validator = call_mul_device_kernel(self.only_gpu, \
                self.cpu_device_env, self.gpu_device_env, \
                self.global_size, self.A, x)
        self.assertTrue(np.all(self.A == validator))

        x = np.float64(3.0)

        validator = call_mul_device_kernel(self.only_gpu, \
                self.cpu_device_env, self.gpu_device_env, \
                self.global_size, self.A, x)
        self.assertTrue(np.all(self.A == validator))

    def test_bool_arg(self):
        @dppy.kernel
        def check_bool_kernel(A, test):
            if test:
                A[0] = 111
            else:
                A[0] = 222

        A = np.array([0], dtype='float64')

        if self.gpu_device_env:
            check_bool_kernel[self.gpu_device_env, self.global_size](A, True)
            self.assertTrue(A[0] == 111)
            check_bool_kernel[self.gpu_device_env, self.global_size](A, False)
            self.assertTrue(A[0] == 222)

if __name__ == '__main__':
    unittest.main()
