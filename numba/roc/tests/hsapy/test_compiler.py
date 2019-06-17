from __future__ import print_function, absolute_import

import tempfile
import os
import numpy as np

import numba.unittest_support as unittest
from numba import roc
from numba import types
from numba.roc import compiler
from numba.roc.hsadrv.driver import hsa as hsart
from numba.roc.hsadrv.driver import BrigModule, Executable, Program


def copy_kernel(out, inp):
    out[0] = inp[0]


def copy_kernel_1d(out, inp):
    i = roc.get_global_id(0)
    if i < out.size:
        out[i] = inp[i]


def assign_value(out, inp):
    i = roc.get_global_id(0)
    if i < out.size:
        out[i] = inp


class TestCodeGeneration(unittest.TestCase):
    def test_copy_kernel(self):
        arytype = types.float32[:]
        kernel = compiler.compile_kernel(copy_kernel, [arytype] * 2)
        self.assertIn(".globl\t{0}".format(kernel.entry_name),
                      kernel.assembly)

    def test_copy_kernel_1d(self):
        arytype = types.float32[:]
        kernel = compiler.compile_kernel(copy_kernel_1d, [arytype] * 2)
        self.assertIn(".globl\t{0}".format(kernel.entry_name),
                      kernel.assembly)


class _TestBase(unittest.TestCase):
    def setUp(self):
        self.gpu = [a for a in hsart.agents if a.is_component][0]
        self.cpu = [a for a in hsart.agents if not a.is_component][0]
        self.queue = self.gpu.create_queue_multi(self.gpu.queue_max_size)

    def tearDown(self):
        del self.queue
        del self.gpu
        del self.cpu


class TestExecution(unittest.TestCase):
    def test_hsa_kernel(self):
        src = np.arange(1024, dtype=np.float32)
        dst = np.zeros_like(src)

        # Compiler kernel
        arytype = types.float32[::1]
        kernel = compiler.compile_kernel(copy_kernel_1d, [arytype] * 2)

        # Run kernel
        kernel[src.size // 256, 256](dst, src)

        np.testing.assert_equal(src, dst)


class TestKernelArgument(unittest.TestCase):
    def _test_template(self, nbtype, src):
        dtype = np.dtype(str(nbtype))
        dst = np.zeros(1, dtype=dtype)
        src = dtype.type(src)
        arytype = nbtype[::1]
        kernel = compiler.compile_kernel(assign_value, [arytype, nbtype])
        kernel[1, 1](dst, src)
        self.assertEqual(dst[0], src)

    def test_float64(self):
        self._test_template(nbtype=types.float64, src=1. / 3.)

    def test_float32(self):
        self._test_template(nbtype=types.float32, src=1. / 3.)

    def test_int32(self):
        self._test_template(nbtype=types.int32, src=123)

    def test_int16(self):
        self._test_template(nbtype=types.int16, src=123)

    def test_complex64(self):
        self._test_template(nbtype=types.complex64, src=12 + 34j)

    def test_complex128(self):
        self._test_template(nbtype=types.complex128, src=12 + 34j)


def udt_devfunc(a, i):
    return a[i]


class TestDeviceFunction(unittest.TestCase):
    def test_device_function(self):
        src = np.arange(10, dtype=np.int32)
        dst = np.zeros_like(src)

        arytype = types.int32[::1]
        devfn = compiler.compile_device(udt_devfunc, arytype.dtype,
                                        [arytype, types.intp])

        def udt_devfunc_caller(dst, src):
            i = roc.get_global_id(0)
            if i < dst.size:
                dst[i] = devfn(src, i)

        kernel = compiler.compile_kernel(udt_devfunc_caller,
                                         [arytype, arytype])

        kernel[src.size, 1](dst, src)
        np.testing.assert_equal(dst, src)


if __name__ == '__main__':
    unittest.main()

