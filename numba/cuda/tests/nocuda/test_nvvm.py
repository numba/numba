from __future__ import absolute_import, print_function, division

from numba.cuda.compiler import compile_kernel
from numba.cuda.cudadrv import nvvm
from numba.cuda.testing import skip_on_cudasim, SerialMixin
from numba import unittest_support as unittest
from numba import types, utils


@skip_on_cudasim('libNVVM not supported in simulator')
@unittest.skipIf(utils.MACHINE_BITS == 32, "CUDA not support for 32-bit")
@unittest.skipIf(not nvvm.is_available(), "No libNVVM")
class TestNvvmWithoutCuda(SerialMixin, unittest.TestCase):
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

    def test_nvvm_memset_fixup(self):
        """
        Test llvm.memset changes in llvm7.
        In LLVM7 the alignment parameter can be implicitly provided as
        an attribute to pointer in the first argument.
        """
        def foo(x):
            # Triggers a generation of llvm.memset
            for i in range(x.size):
                x[i] = 0

        cukern = compile_kernel(foo, args=(types.int32[::1],), link=())
        original = cukern._func.ptx.llvmir
        self.assertIn("call void @llvm.memset", original)
        fixed = nvvm.llvm39_to_34_ir(original)
        self.assertIn("call void @llvm.memset", fixed)
        # Check original IR
        for ln in original.splitlines():
            if 'call void @llvm.memset' in ln:
                # Missing i32 4 in the 2nd last argument
                self.assertRegexpMatches(
                    ln,
                    r'i64 %\d+, i1 false\)'.replace(' ', r'\s+'),
                )
        # Check fixed IR
        for ln in fixed.splitlines():
            if 'call void @llvm.memset' in ln:
                # The i32 4 is the alignment
                self.assertRegexpMatches(
                    ln,
                    r'i32 4, i1 false\)'.replace(' ', r'\s+'),
                )


if __name__ == '__main__':
    unittest.main()
