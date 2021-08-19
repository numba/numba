from numba.cuda.cudadrv import nvvm
from numba.cuda.testing import skip_on_cudasim
from numba.core import utils

from llvmlite import ir
from llvmlite import binding as llvm

import unittest


original = "call void @llvm.memset.p0i8.i64(" \
           "i8* align 4 %arg.x.41, i8 0, i64 %0, i1 false)"

missing_align = "call void @llvm.memset.p0i8.i64(" \
                "i8* %arg.x.41, i8 0, i64 %0, i1 false)"


@skip_on_cudasim('libNVVM not supported in simulator')
@unittest.skipIf(utils.MACHINE_BITS == 32, "CUDA not support for 32-bit")
@unittest.skipIf(not nvvm.is_available(), "No libNVVM")
class TestNvvmWithoutCuda(unittest.TestCase):
    def test_nvvm_memset_fixup_for_34(self):
        """
        Test llvm.memset changes in llvm7.
        In LLVM7 the alignment parameter can be implicitly provided as
        an attribute to pointer in the first argument.
        """
        fixed = nvvm.llvm100_to_34_ir(original)
        self.assertIn("call void @llvm.memset", fixed)

        for ln in fixed.splitlines():
            if 'call void @llvm.memset' in ln:
                # The i32 4 is the alignment
                self.assertRegexpMatches(
                    ln,
                    r'i32 4, i1 false\)'.replace(' ', r'\s+'),
                )

    def test_nvvm_memset_fixup_for_34_missing_align(self):
        """
        We require alignment to be specified as a parameter attribute to the
        dest argument of a memset.
        """
        with self.assertRaises(ValueError) as e:
            nvvm.llvm100_to_34_ir(missing_align)

        self.assertIn(str(e.exception),
                      "No alignment attribute found on memset dest")

    def test_nvvm_accepts_encoding(self):
        # Test that NVVM will accept a constant containing all possible 8-bit
        # characters. Taken from the test case added in llvmlite PR #53:
        #
        #     https://github.com/numba/llvmlite/pull/53
        #
        # This test case is included in Numba to ensure that the encoding used
        # by llvmlite (e.g. utf-8, latin1, etc.) does not result in an input to
        # NVVM that it cannot parse correctly

        # Create a module with a constant containing all 8-bit characters
        c = ir.Constant(ir.ArrayType(ir.IntType(8), 256),
                        bytearray(range(256)))
        m = ir.Module()
        gv = ir.GlobalVariable(m, c.type, "myconstant")
        gv.global_constant = True
        gv.initializer = c
        nvvm.fix_data_layout(m)

        # Parse with LLVM then dump the parsed module into NVVM
        parsed = llvm.parse_assembly(str(m))
        ptx = nvvm.llvm_to_ptx(str(parsed))

        # Ensure all characters appear in the generated constant array.
        elements = ", ".join([str(i) for i in range(256)])
        myconstant = f"myconstant[256] = {{{elements}}}".encode('utf-8')
        self.assertIn(myconstant, ptx)


if __name__ == '__main__':
    unittest.main()
