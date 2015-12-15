from __future__ import print_function, absolute_import

import numba.unittest_support as unittest
from numba.hsa.hlc import hlc
from numba.hsa.hsadrv import enums

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
        # Generate BRIG
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

        nelem = 1

        sig = hsa.create_signal(nelem)

        src = np.random.random(nelem).astype(np.float32)
        dst = np.zeros_like(src)


        def dgpu_count():
            """
            Returns the number of discrete GPUs present on the current machine.
            """
            known_dgpus = frozenset([b'Fiji'])
            known_apus = frozenset([b'Spectre'])
            known_cpus = frozenset([b'Kaveri'])

            ngpus = 0
            for a in hsa.agents:
                name = getattr(a, "name").lower()
                for g in known_dgpus:
                    if g.lower() in name:
                        ngpus += 1
            return ngpus


        if(dgpu_count() > 0):
            gpu = [a for a in hsa.agents if a.is_component][0]
            cpu = [a for a in hsa.agents if not a.is_component][0]

            gpu_regions = gpu.regions
            gpu_only_coarse_regions = list()
            gpu_host_accessible_coarse_regions = list()
            for r in gpu_regions:
                if r.supports(enums.HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED):
                    if r.host_accessible:
                        gpu_host_accessible_coarse_regions.append(r)
                    else:
                        gpu_only_coarse_regions.append(r)
            # check we have 1+ coarse gpu region(s) of each type
            self.assertGreater(len(gpu_only_coarse_regions), 0)
            self.assertGreater(len(gpu_host_accessible_coarse_regions), 0)

            # alloc host accessible memory
            gpu_host_accessible_region = gpu_host_accessible_coarse_regions[0]
            host_in_ptr = gpu_host_accessible_region.allocate(ctypes.c_float *
                                                                nelem)
            self.assertNotEqual(ctypes.addressof(host_in_ptr), 0,
                    "pointer must not be NULL")
            host_out_ptr = gpu_host_accessible_region.allocate(ctypes.c_float *
                                                                nelem)
            self.assertNotEqual(ctypes.addressof(host_out_ptr), 0,
                    "pointer must not be NULL")

            # init mem with data
            hsa.hsa_memory_copy(host_in_ptr, src.ctypes.data, src.nbytes)
            hsa.hsa_memory_copy(host_out_ptr, dst.ctypes.data, dst.nbytes)

            # alloc gpu only memory
            gpu_only_region = gpu_only_coarse_regions[0]
            gpu_in_ptr = gpu_only_region.allocate(ctypes.c_float * nelem)
            self.assertNotEqual(ctypes.addressof(gpu_in_ptr), 0,
                    "pointer must not be NULL")
            gpu_out_ptr = gpu_only_region.allocate(ctypes.c_float * nelem)
            self.assertNotEqual(ctypes.addressof(gpu_out_ptr), 0,
                    "pointer must not be NULL")

            # copy memory from host accessible location to gpu only
            hsa.hsa_memory_copy(gpu_in_ptr, host_in_ptr, src.nbytes)

             # Do kernargs


            # Find a coarse region (for better performance on dGPU) in which
            # to place kernargs. NOTE: This violates the HSA spec
            kernarg_regions = list()
            for r in gpu_host_accessible_coarse_regions:
               # NOTE: VIOLATION
               # if r.supports(enums.HSA_REGION_GLOBAL_FLAG_KERNARG):
               kernarg_regions.append(r)
            self.assertGreater(len(kernarg_regions), 0)

            kernarg_region = kernarg_regions[0]

            kernarg_ptr = kernarg_region.allocate(2 * ctypes.c_void_p)
                                                   # ^- that is not nice
            self.assertNotEqual(ctypes.addressof(kernarg_ptr), 0,
                    "pointer must not be NULL")

            # wire in gpu memory
            kernarg_ptr[0] = ctypes.addressof(gpu_in_ptr)
            kernarg_ptr[1] = ctypes.addressof(gpu_out_ptr)

            kernargs = kernarg_ptr

        else:
            kernarg_region = [r for r in agent.regions
                    if r.supports(enums.HSA_REGION_GLOBAL_FLAG_KERNARG)][0]

            kernarg_types = (ctypes.c_void_p * 2)
            kernargs = kernarg_region.allocate(kernarg_types)

            kernargs[0] = src.ctypes.data
            kernargs[1] = dst.ctypes.data

            hsa.hsa_memory_register(src.ctypes.data, src.nbytes)
            hsa.hsa_memory_register(dst.ctypes.data, dst.nbytes)
            hsa.hsa_memory_register(ctypes.byref(kernargs),
                                ctypes.sizeof(kernargs))

        queue = agent.create_queue_single(32)
        queue.dispatch(sym, kernargs, workgroup_size=(1,),
                       grid_size=(1,))

        if(dgpu_count() > 0):
            # copy result back to host accessible memory to check
            hsa.hsa_memory_copy(host_out_ptr, gpu_out_ptr, src.nbytes)
            hsa.hsa_memory_copy(dst.ctypes.data, host_out_ptr, src.nbytes)

        np.testing.assert_equal(dst, src)

        if(dgpu_count() > 0):
            # free
            hsa.hsa_memory_free(host_in_ptr)
            hsa.hsa_memory_free(host_out_ptr)
            hsa.hsa_memory_free(gpu_in_ptr)
            hsa.hsa_memory_free(gpu_out_ptr)

if __name__ == '__main__':
    unittest.main()

