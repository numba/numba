#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
from numba import dppl
import dpctl
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase
import math

@dppl.kernel
def dppl_fabs(a,b):
    i = dppl.get_global_id(0)
    b[i] = math.fabs(a[i])

@dppl.kernel
def dppl_exp(a,b):
    i = dppl.get_global_id(0)
    b[i] = math.exp(a[i])

@dppl.kernel
def dppl_log(a,b):
    i = dppl.get_global_id(0)
    b[i] = math.log(a[i])

@dppl.kernel
def dppl_sqrt(a,b):
    i = dppl.get_global_id(0)
    b[i] = math.sqrt(a[i])

@dppl.kernel
def dppl_sin(a,b):
    i = dppl.get_global_id(0)
    b[i] = math.sin(a[i])

@dppl.kernel
def dppl_cos(a,b):
    i = dppl.get_global_id(0)
    b[i] = math.cos(a[i])

@dppl.kernel
def dppl_tan(a,b):
    i = dppl.get_global_id(0)
    b[i] = math.tan(a[i])

global_size = 10
N = global_size

a = np.array(np.random.random(N), dtype=np.float32)

def driver(a, jitfunc):
    b = np.ones_like(a)
    # Device buffers
    jitfunc[global_size, dppl.DEFAULT_LOCAL_SIZE](a, b)
    return b


def test_driver(input_arr, device_ty, jitfunc):
    out_actual = None
    if device_ty == "GPU":
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            out_actual = driver(input_arr, jitfunc)
    elif device_ty == "CPU":
        with dpctl.device_context("opencl:cpu") as cpu_queue:
            out_actual = driver(input_arr, jitfunc)
    else:
        print("Unknown device type")
        raise SystemExit()

    return out_actual


@unittest.skipUnless(dpctl.has_cpu_queues(), 'test only on CPU system')
class TestDPPLMathFunctionsCPU(DPPLTestCase):
    def test_fabs_cpu(self):
        b_actual = test_driver(a, "CPU", dppl_fabs)
        b_expected = np.fabs(a)
        self.assertTrue(np.all(b_actual == b_expected))

    def test_sin_cpu(self):
        b_actual = test_driver(a, "CPU", dppl_sin)
        b_expected = np.sin(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_cos_cpu(self):
        b_actual = test_driver(a, "CPU", dppl_cos)
        b_expected = np.cos(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_exp_cpu(self):
        b_actual = test_driver(a, "CPU", dppl_exp)
        b_expected = np.exp(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_sqrt_cpu(self):
        b_actual = test_driver(a, "CPU", dppl_sqrt)
        b_expected = np.sqrt(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_log_cpu(self):
        b_actual = test_driver(a, "CPU", dppl_log)
        b_expected = np.log(a)
        self.assertTrue(np.allclose(b_actual,b_expected))


@unittest.skipUnless(dpctl.has_gpu_queues(), 'test only on GPU system')
class TestDPPLMathFunctionsGPU(DPPLTestCase):
    def test_fabs_gpu(self):
        b_actual = test_driver(a, "GPU", dppl_fabs)
        b_expected = np.fabs(a)
        self.assertTrue(np.all(b_actual == b_expected))

    def test_sin_gpu(self):
        b_actual = test_driver(a, "GPU", dppl_sin)
        b_expected = np.sin(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_cos_gpu(self):
        b_actual = test_driver(a, "GPU", dppl_cos)
        b_expected = np.cos(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_exp_gpu(self):
        b_actual = test_driver(a, "GPU", dppl_exp)
        b_expected = np.exp(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_sqrt_gpu(self):
        b_actual = test_driver(a, "GPU", dppl_sqrt)
        b_expected = np.sqrt(a)
        self.assertTrue(np.allclose(b_actual,b_expected))

    def test_log_gpu(self):
        b_actual = test_driver(a, "GPU", dppl_log)
        b_expected = np.log(a)
        self.assertTrue(np.allclose(b_actual,b_expected))


if __name__ == '__main__':
    unittest.main()
