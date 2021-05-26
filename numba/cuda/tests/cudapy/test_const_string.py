import re
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from llvmlite import ir


@skip_on_cudasim("This is testing CUDA backend code generation")
class TestConstStringCodegen(unittest.TestCase):
    def test_const_string(self):
        # These imports are incompatible with CUDASIM
        from numba.cuda.descriptor import cuda_target
        from numba.cuda.cudadrv.nvvm import llvm_to_ptx, ADDRSPACE_CONSTANT

        targetctx = cuda_target.target_context
        mod = targetctx.create_module("")
        textstring = 'A Little Brown Fox'
        gv0 = targetctx.insert_const_string(mod, textstring)
        # Insert the same const string a second time - the first should be
        # reused.
        targetctx.insert_const_string(mod, textstring)

        res = re.findall(r"@\"__conststring__.*internal.*constant.*\["
                         r"19\s+x\s+i8\]", str(mod))
        # Ensure that the const string was only inserted once
        self.assertEqual(len(res), 1)

        fnty = ir.FunctionType(ir.IntType(8).as_pointer(), [])

        # Using insert_const_string
        fn = ir.Function(mod, fnty, "test_insert_const_string")
        builder = ir.IRBuilder(fn.append_basic_block())
        res = targetctx.insert_addrspace_conv(builder, gv0,
                                              addrspace=ADDRSPACE_CONSTANT)
        builder.ret(res)

        matches = re.findall(r"@\"__conststring__.*internal.*constant.*\["
                             r"19\s+x\s+i8\]", str(mod))
        self.assertEqual(len(matches), 1)

        # Using insert_string_const_addrspace
        fn = ir.Function(mod, fnty, "test_insert_string_const_addrspace")
        builder = ir.IRBuilder(fn.append_basic_block())
        res = targetctx.insert_string_const_addrspace(builder, textstring)
        builder.ret(res)

        matches = re.findall(r"@\"__conststring__.*internal.*constant.*\["
                             r"19\s+x\s+i8\]", str(mod))
        self.assertEqual(len(matches), 1)

        ptx = llvm_to_ptx(str(mod)).decode('ascii')
        matches = list(re.findall(r"\.const.*__conststring__", ptx))

        self.assertEqual(len(matches), 1)


class TestConstString(CUDATestCase):
    def test_assign_const_string(self):
        # Inspired by the reproducer from Issue #7041.

        @cuda.jit
        def str_assign(arr):
            i = cuda.grid(1)
            if i < len(arr):
                arr[i] = "XYZ"

        n_strings = 8
        arr = np.zeros(n_strings + 1, dtype="<U12")
        str_assign[1, n_strings](arr)

        # Expected result, e.g.:
        #     ['XYZ' 'XYZ' 'XYZ' 'XYZ' 'XYZ' 'XYZ' 'XYZ' 'XYZ' '']
        expected = np.zeros_like(arr)
        expected[:-1] = 'XYZ'

        np.testing.assert_equal(arr, expected)


if __name__ == '__main__':
    unittest.main()
