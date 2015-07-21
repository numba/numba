from __future__ import absolute_import, print_function, division

from numba.cuda.compiler import compile_kernel
from numba.cuda.cudadrv import nvvm
from numba.cuda.testing import skip_on_cudasim
from numba import unittest_support as unittest
from numba import types, utils


@skip_on_cudasim('libNVVM not supported in simulator')
@unittest.skipIf(utils.MACHINE_BITS == 32, "CUDA not support for 32-bit")
@unittest.skipIf(not nvvm.is_available(), "No libNVVM")
class TestNvvmWithoutCuda(unittest.TestCase):
    def test_nvvm_llvm_to_ptx(self):
        """
        A simple test to exercise nvvm.llvm_to_ptx()
        to trigger issues with mismatch NVVM API.
        """

        def foo(x):
            x[0] = 123

        cukern = compile_kernel(foo, args=(types.int32[::1],), link=())
        llvmir = cukern._func.ptx.llvmir
        ptx = nvvm.llvm_to_ptx(llvmir)
        self.assertIn("foo", ptx.decode('ascii'))


if __name__ == '__main__':
    unittest.main()
