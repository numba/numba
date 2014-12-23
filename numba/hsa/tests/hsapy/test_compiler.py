from __future__ import print_function, absolute_import

import numba.unittest_support as unittest
from numba.hsa import compiler
from numba import hsa
from numba import types


def copy_kernel(out, inp):
    out[0] = inp[0]


def copy_kernel_1d(out, inp):
    i = hsa.get_global_id(1)
    if i < out.size:
        out[i] = inp[i]


class TestCompiler(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()

