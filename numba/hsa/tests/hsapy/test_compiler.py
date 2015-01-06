from __future__ import print_function, absolute_import

import tempfile
import os
import ctypes
from time import time

import numba.unittest_support as unittest
from numba import hsa
from numba import types
from numba.hsa import compiler
from numba.hsa.hsadrv.driver import hsa as hsart, BrigModule
from numba.targets.arrayobj import make_array_ctype


def copy_kernel(out, inp):
    out[0] = inp[0]


def copy_kernel_1d(out, inp):
    i = hsa.get_global_id(0)
    if i < out.size:
        out[i] = inp[i]


def assign_value(out, inp):
    i = hsa.get_global_id(0)
    if i < out.size:
        out[i] = inp


class TestCodeGeneration(unittest.TestCase):
    def test_copy_kernel(self):
        arytype = types.float32[:]
        kernel = compiler.compile_kernel(copy_kernel, [arytype] * 2)
        self.assertIn("prog kernel &{0}".format(kernel.entry_name),
                      kernel.assembly)

    def test_copy_kernel_1d(self):
        arytype = types.float32[:]
        kernel = compiler.compile_kernel(copy_kernel_1d, [arytype] * 2)
        self.assertIn("prog kernel &{0}".format(kernel.entry_name),
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


class TestCodeLoading(_TestBase):
    def test_loading_from_file(self):
        arytype = types.float32[:]
        kernel = compiler.compile_kernel(copy_kernel_1d, [arytype] * 2)

        # Write the brig file out
        brig_file = tempfile.NamedTemporaryFile(delete=False)
        with brig_file:
            brig_file.write(kernel.binary)

        # Load BRIG file
        symbol = '&{0}'.format(kernel.entry_name)
        brig_module = BrigModule.from_file(brig_file.name)
        symbol_offset = brig_module.find_symbol_offset(symbol)
        self.assertTrue(symbol_offset)
        program = hsart.create_program([self.gpu])
        module = program.add_module(brig_module)
        code_descriptor = program.finalize(self.gpu, module, symbol_offset)
        self.assertGreater(code_descriptor._id.kernarg_segment_byte_size, 0)

        # Cleanup
        os.unlink(brig_file.name)

    def test_loading_from_memory(self):
        arytype = types.float32[:]
        kernel = compiler.compile_kernel(copy_kernel_1d, [arytype] * 2)

        # Load BRIG memory
        symbol = '&{0}'.format(kernel.entry_name)
        brig_module = BrigModule.from_memory(kernel.binary)
        symbol_offset = brig_module.find_symbol_offset(symbol)
        self.assertTrue(symbol_offset)
        program = hsart.create_program([self.gpu])
        module = program.add_module(brig_module)
        code_descriptor = program.finalize(self.gpu, module, symbol_offset)
        self.assertGreater(code_descriptor._id.kernarg_segment_byte_size, 0)


import numpy as np
from numba.hsa.hsadrv import enums


class TestExecution(unittest.TestCase):
    def test_execute(self):
        src = np.arange(1024 * 1024 * 20, dtype=np.float32)
        dst = np.zeros_like(src)

        components = [a for a in hsart.agents if a.is_component]

        gpu = components[0]
        print("Using agent: {0} with queue size: {1}".format(gpu.name,
                                                             gpu.queue_max_size))
        q = gpu.create_queue_multi(gpu.queue_max_size)

        # Compiler kernel
        arytype = types.float32[::1]
        kernel = compiler.compile_kernel(copy_kernel_1d, [arytype] * 2)
        # print(kernel.assembly)

        # Load BRIG memory
        symbol = '&{0}'.format(kernel.entry_name)
        brig_module = BrigModule.from_memory(kernel.binary)
        symbol_offset = brig_module.find_symbol_offset(symbol)
        program = hsart.create_program([gpu])
        module = program.add_module(brig_module)
        code_descriptor = program.finalize(gpu, module, symbol_offset)

        print(program)

        kernarg_regions = [r for r in gpu.regions if r.supports_kernargs]
        assert kernarg_regions
        kernarg_region = kernarg_regions[0]
        # Specify the argument types required by the kernel and allocate them
        # note: in an ideal world this should come from kernel metadata
        kernarg_types = ctypes.c_void_p * (6 + 2)
        kernargs = kernarg_region.allocate(kernarg_types)

        cstruct = make_array_ctype(ndim=1)

        src_cstruct = cstruct(parent=None,
                              data=src.ctypes.data,
                              shape=src.ctypes.shape,
                              strides=src.ctypes.strides)

        dst_cstruct = cstruct(parent=None,
                              data=dst.ctypes.data,
                              shape=dst.ctypes.shape,
                              strides=dst.ctypes.strides)

        kernargs[0] = 0
        kernargs[1] = 0
        kernargs[2] = 0
        kernargs[3] = 0
        kernargs[4] = 0
        kernargs[5] = 0
        kernargs[6] = ctypes.addressof(dst_cstruct)
        kernargs[7] = ctypes.addressof(src_cstruct)

        hsart.hsa_memory_register(kernargs[6], ctypes.sizeof(cstruct))
        hsart.hsa_memory_register(kernargs[7], ctypes.sizeof(cstruct))
        hsart.hsa_memory_register(src.ctypes.data, src.nbytes)
        hsart.hsa_memory_register(dst.ctypes.data, dst.nbytes)
        hsart.hsa_memory_register(ctypes.byref(kernargs),
                                  ctypes.sizeof(kernargs))

        # sync (in fact, dispatch will create a dummy signal for the dispatch and
        # wait for it before returning)
        print("dispatch synchronous... ", end="")
        t_start = time()
        q.dispatch(code_descriptor, kernargs, workgroup_size=(256,),
                   grid_size=(src.size,))
        t_end = time()
        print("ellapsed: {0:10.9f} s.".format(t_end - t_start))

        # async: handle the signal by hand
        print("dispatch asynchronous... ", end="")
        t_start = time()
        s = hsart.create_signal(1)
        q.dispatch(code_descriptor, kernargs,
                   workgroup_size=(256,), grid_size=(src.size,), signal=s)
        t_launched = time()
        hsart.hsa_signal_wait_acquire(s._id, enums.HSA_LT, 1, -1,
                                      enums.HSA_WAIT_EXPECTANCY_UNKNOWN)
        t_end = time()
        print("launch: {0:10.9f} s. total: {1:10.9g} s.".format(
            t_launched - t_start, t_end - t_start))

        # this is placed in the kernarg_region for symmetry, but shouldn't be required.
        kernarg_region.free(kernargs)

        np.testing.assert_equal(src, dst)

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

        arytype = nbtype[::1]
        kernel = compiler.compile_kernel(assign_value, [arytype, nbtype])
        print(kernel.assembly)
        print(dst, src)
        kernel[1, 1](dst, src)

        print(dst, src)
        self.assertEqual(dst[0], src)

    def test_float64(self):
        self._test_template(nbtype=types.float64, src=1. / 3.)


if __name__ == '__main__':
    unittest.main()

