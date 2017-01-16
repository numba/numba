from __future__ import print_function

import re
from numba.cuda.testing import unittest, skip_on_cudasim
from llvmlite import ir


@skip_on_cudasim("This is testing CUDA backend code generation")
class TestCudaConstString(unittest.TestCase):
    def test_const_string(self):
        # These imports is incompatible with CUDASIM
        from numba.cuda.descriptor import CUDATargetDesc
        from numba.cuda.cudadrv.nvvm import llvm_to_ptx, ADDRSPACE_CONSTANT

        targetctx = CUDATargetDesc.targetctx
        codegen = targetctx.codegen()
        library = codegen.create_library(name='test_const_string')
        mod = library.create_ir_module(name='test_const_string')
        textstring = 'A Little Brown Fox'
        gv0 = targetctx.insert_const_string(mod, textstring)
        gv1 = targetctx.insert_const_string(mod, textstring)

        res = re.findall(r"@\"__conststring__.*internal.*constant.*\["
                         r"19\s+x\s+i8\]", str(mod))
        self.assertEqual(len(res), 1)

        fnty = ir.FunctionType(ir.IntType(8).as_pointer(), [])

        # Using insert_const_string
        fn = mod.add_function(fnty, name="test_insert_const_string")
        builder = ir.IRBuilder(fn.append_basic_block())
        res = targetctx.insert_addrspace_conv(builder, gv0,
                                              addrspace=ADDRSPACE_CONSTANT)
        builder.ret(res)

        matches = re.findall(r"@\"__conststring__.*internal.*constant.*\["
                             r"19\s+x\s+i8\]", str(mod))
        self.assertEqual(len(matches), 1)

        # Using insert_string_const_addrspace
        fn = mod.add_function(fnty, name="test_insert_string_const_addrspace")
        builder = ir.IRBuilder(fn.append_basic_block())
        res = targetctx.insert_string_const_addrspace(builder, textstring)
        builder.ret(res)

        matches = re.findall(r"@\"__conststring__.*internal.*constant.*\["
                             r"19\s+x\s+i8\]", str(mod))
        self.assertEqual(len(matches), 1)

        library.add_ir_module(mod)
        library.finalize()


        ptx = llvm_to_ptx(str(library._final_module)).decode('ascii')
        matches = list(re.findall(r"\.const.*__conststring__", ptx))

        self.assertEqual(len(matches), 1)


if __name__ == '__main__':
    unittest.main()
