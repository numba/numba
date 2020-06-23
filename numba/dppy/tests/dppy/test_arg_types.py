from __future__ import print_function, division, absolute_import

import numpy as np

from numba import dppy
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase
import dppy as ocldrv


@dppy.kernel
def mul_kernel(A, B, test):
    i = dppy.get_global_id(0)
    B[i] = A[i] * test

def call_mul_device_kernel(global_size, A, B, test):
    mul_kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](A, B, test)


global_size = 10
N = global_size
A = np.array(np.random.random(N), dtype=np.float32)
B = np.array(np.random.random(N), dtype=np.float32)


@unittest.skipUnless(dppy.has_cpu_device, 'test only on CPU system')
class TestDPPYArrayArgCPU(DPPYTestCase):
    def test_integer_arg(self):
        x = np.int32(2)
        with ocldrv.cpu_context(0) as device_env:
            call_mul_device_kernel(global_size, A, B, x)
        self.assertTrue(np.all((A * x) == B))

    def test_float_arg(self):
        x = np.float32(2.0)
        with ocldrv.cpu_context(0) as device_env:
            call_mul_device_kernel(global_size, A, B, x)
            self.assertTrue(np.all(A * x == B))

            x = np.float64(3.0)
            call_mul_device_kernel(global_size, A, B, x)
            self.assertTrue(np.all(A * x == B))

    def test_bool_arg(self):
        @dppy.kernel
        def check_bool_kernel(A, test):
            if test:
                A[0] = 111
            else:
                A[0] = 222

        A = np.array([0], dtype='float64')

        with ocldrv.cpu_context(0) as device_env:
            check_bool_kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](A, True)
            self.assertTrue(A[0] == 111)
            check_bool_kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](A, False)
            self.assertTrue(A[0] == 222)


@unittest.skipUnless(dppy.has_gpu_device, 'test only on GPU system')
class TestDPPYArrayArgGPU(DPPYTestCase):
    def test_integer_arg(self):
        x = np.int32(2)
        with ocldrv.igpu_context(0) as device_env:
            call_mul_device_kernel(global_size, A, B, x)
        self.assertTrue(np.all((A * x) == B))

    def test_float_arg(self):
        x = np.float32(2.0)
        with ocldrv.igpu_context(0) as device_env:
            call_mul_device_kernel(global_size, A, B, x)
            self.assertTrue(np.all(A * x == B))

            x = np.float64(3.0)
            call_mul_device_kernel(global_size, A, B, x)
            self.assertTrue(np.all(A * x == B))

    def test_bool_arg(self):
        @dppy.kernel
        def check_bool_kernel(A, test):
            if test:
                A[0] = 111
            else:
                A[0] = 222

        A = np.array([0], dtype='float64')

        with ocldrv.igpu_context(0) as device_env:
            check_bool_kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](A, True)
            self.assertTrue(A[0] == 111)
            check_bool_kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](A, False)
            self.assertTrue(A[0] == 222)

if __name__ == '__main__':
    unittest.main()
