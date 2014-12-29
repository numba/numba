from __future__ import print_function, absolute_import

import tempfile
import os

import numba.unittest_support as unittest
from numba import hsa
from numba import types
from numba.hsa import compiler
from numba.hsa.hsadrv.driver import hsa as hsart, Queue, BrigModule


def copy_kernel(out, inp):
    out[0] = inp[0]


def copy_kernel_1d(out, inp):
    i = hsa.get_global_id(1)
    if i < out.size:
        out[i] = inp[i]


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
    def test_copy_kernel_1d(self):
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


if __name__ == '__main__':
    unittest.main()

