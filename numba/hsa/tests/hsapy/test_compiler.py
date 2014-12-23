from __future__ import print_function, absolute_import

import numba.unittest_support as unittest
from numba.hsa import compiler
from numba import types


def copy_kernel(out, inp):
    out[0] = inp[0]


class TestCompiler(unittest.TestCase):
    def test_add(self):
        arytype = types.float32[:]
        kernel = compiler.compile_kernel(copy_kernel, [arytype] * 2)
        print(kernel)


if __name__ == '__main__':
    unittest.main()

