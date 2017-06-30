from __future__ import print_function, absolute_import
from numba import unittest_support as unittest
from numba.ocl.ocldrv.driver import driver as cl
from numba.ocl.ocldrv.devices import _runtime as rt
from numba.ocl.ocldrv import spirv
from numba.ocl.ocldrv import spir2


sample_spir = """
; ModuleID = 'kernel.out.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @square(float addrspace(1)* nocapture %input, float addrspace(1)* nocapture %output) #0 {
  %1 = load float, float addrspace(1)* %input, align 4, !tbaa !8
  store float %1, float addrspace(1)* %output, align 4, !tbaa !8
  ret void
}

attributes #0 = { nounwind }

!opencl.kernels = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!6}
!opencl.used.extensions = !{!7}
!opencl.used.optional.core.features = !{!7}
!opencl.compiler.options = !{!7}

!0 = !{void (float addrspace(1)*, float addrspace(1)*)* @square, !1, !2, !3, !4, !5}
!1 = !{!"kernel_arg_addr_space", i32 1, i32 1}
!2 = !{!"kernel_arg_access_qual", !"none", !"none"}
!3 = !{!"kernel_arg_type", !"float*", !"float*"}
!4 = !{!"kernel_arg_type_qual", !"", !""}
!5 = !{!"kernel_arg_base_type", !"float*", !"float*"}
!6 = !{i32 2, i32 0}
!7 = !{}
!8 = !{!"float", !9}
!9 = !{!"omnipotent char", !10}
!10 = !{!"Simple C/C++ TBAA"}
"""


class TestSPIRLoading(unittest.TestCase):
    def test_spir_loading(self):

        platform = cl.default_platform
        device = cl.default_device
        context = platform.create_context([device])

        if device.opencl_version == (2, 0):
            spir2_bc = spir2.llvm_to_spir2(sample_spir)
            program = context.create_program_from_binary(spir2_bc)
        elif device.opencl_version == (2, 1):
            spirv_bc = spirv.llvm_to_spirv(sample_spir)
            program = context.create_program_from_il(spirv_bc)

        program.build()

        self.assertEqual(len(program.kernel_names), 1)
        self.assertEqual(program.kernel_names[0], "square")


if __name__ == '__main__':
    unittest.main()
