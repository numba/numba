from __future__ import print_function, absolute_import

import numba.unittest_support as unittest
from numba.hsa.hlc import hlc

SPIR_SAMPLE = """
; ModuleID = 'kernel.out.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n32"
target triple = "hsail64-pc-unknown-amdopencl"

define spir_kernel void @copy(float addrspace(1)* nocapture %input,
float addrspace(1)* nocapture %output) {
  %1 = load float addrspace(1)* %input, align 4, !tbaa !8
  store float %1, float addrspace(1)* %output, align 4, !tbaa !8
  ret void
}

!opencl.kernels = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!6}
!opencl.used.extensions = !{!7}
!opencl.used.optional.core.features = !{!7}
!opencl.compiler.options = !{!7}
!0 = metadata !{void (float addrspace(1)*, float addrspace(1)*)* @copy, metadata !1, metadata !2, metadata !3, metadata !4, metadata !5}
!1 = metadata !{metadata !"kernel_arg_addr_space", i32 1, i32 1}
!2 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none"}
!3 = metadata !{metadata !"kernel_arg_type", metadata !"float*", metadata !"float*"}
!4 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !""}
!5 = metadata !{metadata !"kernel_arg_base_type", metadata !"float*", metadata !"float*"}
!6 = metadata !{i32 1, i32 2}
!7 = metadata !{}
!8 = metadata !{metadata !"float", metadata !9}
!9 = metadata !{metadata !"omnipotent char", metadata !10}
!10 = metadata !{metadata !"Simple C/C++ TBAA"}
"""


class TestHLC(unittest.TestCase):
    def test_hsail(self):
        hlcmod = hlc.Module()
        hlcmod.load_llvm(SPIR_SAMPLE)
        hsail = hlcmod.finalize().hsail
        self.assertIn("prog kernel &copy", hsail)

    def test_brig(self):
        # Genreate BRIG
        hlcmod = hlc.Module()
        hlcmod.load_llvm(SPIR_SAMPLE)
        brig = hlcmod.finalize().brig
        # Check the first 8 bytes for the magic string
        self.assertEqual(brig[:8].decode('latin1'), 'HSA BRIG')

        # Compile
        from numba.hsa.hsadrv.driver import BrigModule, Program, hsa, Executable

        agent = hsa.components[0]
        brigmod = BrigModule(brig)
        prog = Program()
        prog.add_module(brigmod)
        code = prog.finalize(agent.isa)
        ex = Executable()
        ex.load(agent, code)
        ex.freeze()
        sym = ex.get_symbol(agent, "&copy")
        self.assertNotEqual(sym.kernel_object, 0)
        self.assertGreater(sym.kernarg_segment_size, 0)

        # Execute
        import ctypes
        import numpy as np

        sig = hsa.create_signal(1)

        kernarg_region = [r for r in agent.regions if r.supports_kernargs][0]

        kernarg_types = (ctypes.c_void_p * 2)
        kernargs = kernarg_region.allocate(kernarg_types)

        src = np.random.random(1).astype(np.float32)
        dst = np.zeros_like(src)

        kernargs[0] = src.ctypes.data
        kernargs[1] = dst.ctypes.data

        hsa.hsa_memory_register(src.ctypes.data, src.nbytes)
        hsa.hsa_memory_register(dst.ctypes.data, dst.nbytes)
        hsa.hsa_memory_register(ctypes.byref(kernargs),
                                ctypes.sizeof(kernargs))

        queue = agent.create_queue_single(32)
        queue.dispatch(sym, kernargs, workgroup_size=(1,),
                       grid_size=(1,))

        np.testing.assert_equal(dst, src)


if __name__ == '__main__':
    unittest.main()

