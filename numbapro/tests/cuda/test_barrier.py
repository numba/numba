import unittest
import numpy as np
from numba import *
from numbapro import cuda
from numbapro.cudapipeline import nvvm
import support
llvm = '''

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"


define void @always_use_a_barrier(i32* %data) {
entry:
    call void @llvm.nvvm.barrier0()
    ret void
}
    
declare void @llvm.nvvm.barrier0() 

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

!nvvm.annotations = !{!1}
!1 = metadata !{void (i32*)* @always_use_a_barrier, metadata !"kernel", i32 1}

    
'''


@jit(argtypes=[f4[:]], target='gpu')
def cu_array_double(dst):
    smem = cuda.shared.array(shape=(256,), dtype=f4)
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x
    i = tid + blkid * blkdim
    if i == 0: # the first thread only
        for j in range(256):
            smem[j] = dst[j]  # store in smem
    cuda.syncthreads()
    dst[i] = smem[i] * 2      # use smem



class TestCudaBarrier(support.CudaTestCase):
    def test_barrier0(self):
        '''Testing out what llvm.nvvm.barrier0 means.
        The document is useless.
        '''
        ptx = nvvm.llvm_to_ptx(llvm)
        self.assertTrue('bar.sync' in ptx)

    def test_barrier_in_use(self):
        A = np.array(np.random.random(256), dtype=np.float32)
        Gold = A * 2
        
        cu_array_double[(1,), (A.shape[0],)](A)
        
        for i, (got, expect) in enumerate(zip(A, Gold)):
            self.assertEqual(got, expect, "%s != %s at i=%d" % (got, expect, i))

if __name__ == '__main__':
    unittest.main()


