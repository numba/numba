from numba.cuda.compiler import compile_kernel
from numba.cuda.cudadrv import nvvm
from numba.cuda.testing import skip_on_cudasim, SerialMixin
from numba.core import types, utils
import unittest


original = "call void @llvm.memset.p0i8.i64(" \
           "i8* align 4 %arg.x.41, i8 0, i64 %0, i1 false)"

missing_align = "call void @llvm.memset.p0i8.i64(" \
                "i8* %arg.x.41, i8 0, i64 %0, i1 false)"


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
        fixed = nvvm.llvm39_to_34_ir(original)
        self.assertIn("call void @llvm.memset", fixed)

        for ln in fixed.splitlines():
            if 'call void @llvm.memset' in ln:
                # The i32 4 is the alignment
                self.assertRegexpMatches(
                    ln,
                    r'i32 4, i1 false\)'.replace(' ', r'\s+'),
                )

    def test_nvvm_memset_fixup_missing_align(self):
        """
        We require alignment to be specified as a parameter attribute to the
        dest argument of a memset.
        """
        with self.assertRaises(ValueError) as e:
            nvvm.llvm39_to_34_ir(missing_align)

        self.assertIn(str(e.exception),
                      "No alignment attribute found on memset dest")


if __name__ == '__main__':
    unittest.main()
