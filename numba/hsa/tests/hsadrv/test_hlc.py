from __future__ import print_function, absolute_import

import numba.unittest_support as unittest
from numba.hsa.hlc import hlc

SPIR_SAMPLE = """
; ModuleID = 'kernel.out.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n32"
target triple = "spir64-unknown-unknown"

define spir_kernel void @square(float addrspace(1)* nocapture %input, float addrspace(1)* nocapture %output) {
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
!0 = metadata !{void (float addrspace(1)*, float addrspace(1)*)* @square, metadata !1, metadata !2, metadata !3, metadata !4, metadata !5}
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
    def test_hlc(self):
        hlcmod = hlc.Module()
        hlcmod.load_llvm(SPIR_SAMPLE)
        hsail = hlcmod.finalize().hsail
        self.assertIn("prog kernel &square", hsail)


if __name__ == '__main__':
    unittest.main()

