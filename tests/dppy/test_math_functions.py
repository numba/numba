#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppy
import dppy.core as ocldrv
from numba.dppy.testing import unittest
from numba.dppy.testing import DPPYTestCase
import math

@dppy.kernel
def dppy_fabs(a,b):
    i = dppy.get_global_id(0)
    b[i] = math.fabs(a[i])

@dppy.kernel
def dppy_exp(a,b):
    i = dppy.get_global_id(0)
    b[i] = math.exp(a[i])

@dppy.kernel
def dppy_log(a,b):
    i = dppy.get_global_id(0)
    b[i] = math.log(a[i])

@dppy.kernel
def dppy_sqrt(a,b):
    i = dppy.get_global_id(0)
    b[i] = math.sqrt(a[i])

@dppy.kernel
def dppy_sin(a,b):
    i = dppy.get_global_id(0)
    b[i] = math.sin(a[i])

@dppy.kernel
def dppy_cos(a,b):
    i = dppy.get_global_id(0)
    b[i] = math.cos(a[i])

@dppy.kernel
def dppy_tan(a,b):
    i = dppy.get_global_id(0)
    b[i] = math.tan(a[i])

global_size = 10
N = global_size

a = np.array(np.random.random(N), dtype=np.float32)

def driver(a, device_env, jitfunc):
    b = np.ones_like(a)
    # Device buffers
    dA = device_env.copy_array_to_device(a)
    dB = device_env.create_device_array(b)
    jitfunc[global_size, dppy.DEFAULT_LOCAL_SIZE](dA, dB)
    device_env.copy_array_from_device(dB)
    return dB.get_ndarray()


def test_driver(input_arr, device_ty, jitfunc):
    out_actual = None
    if device_ty == "GPU":
        with ocldrv.igpu_context(0) as device_env:
            out_actual = driver(input_arr, device_env, jitfunc)
    elif device_ty == "CPU":
        with ocldrv.cpu_context(0) as device_env:
            out_actual = driver(input_arr, device_env, jitfunc)
    else:
        print("Unknown device type")
        raise SystemExit()

    return out_actual


@unittest.skipUnless(ocldrv.has_cpu_device, 'test only on CPU system')
class TestDPPYMathFunctionsCPU(DPPYTestCase):
    def test_fabs_cpu(self):
        b_actual = test_driver(a, "CPU", dppy_fabs)
        b_expected = np.fabs(a)
        self.assertTrue(np.all(b_actual == b_expected))

    def test_sin_cpu(self):
        b_actual = test_driver(a, "CPU", dppy_sin)
        b_expected = np.sin(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_cos_cpu(self):
        b_actual = test_driver(a, "CPU", dppy_cos)
        b_expected = np.cos(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_exp_cpu(self):
        b_actual = test_driver(a, "CPU", dppy_exp)
        b_expected = np.exp(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_sqrt_cpu(self):
        b_actual = test_driver(a, "CPU", dppy_sqrt)
        b_expected = np.sqrt(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_log_cpu(self):
        b_actual = test_driver(a, "CPU", dppy_log)
        b_expected = np.log(a)
        self.assertTrue(np.allclose(b_actual,b_expected))


@unittest.skipUnless(ocldrv.has_gpu_device, 'test only on GPU system')
class TestDPPYMathFunctionsGPU(DPPYTestCase):
    def test_fabs_gpu(self):
        b_actual = test_driver(a, "GPU", dppy_fabs)
        b_expected = np.fabs(a)
        self.assertTrue(np.all(b_actual == b_expected))

    def test_sin_gpu(self):
        b_actual = test_driver(a, "GPU", dppy_sin)
        b_expected = np.sin(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_cos_gpu(self):
        b_actual = test_driver(a, "GPU", dppy_cos)
        b_expected = np.cos(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_exp_gpu(self):
        b_actual = test_driver(a, "GPU", dppy_exp)
        b_expected = np.exp(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_sqrt_gpu(self):
        b_actual = test_driver(a, "GPU", dppy_sqrt)
        b_expected = np.sqrt(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_log_gpu(self):
        b_actual = test_driver(a, "GPU", dppy_log)
        b_expected = np.log(a)
        self.assertTrue(np.allclose(b_actual,b_expected))


if __name__ == '__main__':
    unittest.main()
