; 
;  Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
; 
;  Please refer to the NVIDIA end user license agreement (EULA) associated
;  with this source code for terms and conditions that govern your use of
;  this software. Any use, reproduction, disclosure, or distribution of
;  this software and related documentation outside the terms of the EULA
;  is strictly prohibited.
; 
; 

; The NVVM IR is equivalent to the following CUDA C code:
;
; __device__ int ave(int a, int b)
; {
;    return (a+b)/2;
; }
;
; __global__ void simple(int *data)
; {
;    int tid = blockIdx.x * blockDim.x + threadIdx.x;
;    data[tid] = ave(tid, tid);
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define i32 @ave(i32 %a, i32 %b) {
entry:
  %add = add nsw i32 %a, %b
  %div = sdiv i32 %add, 2
  ret i32 %div
}

define void @simple(i32* %data) {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul = mul i32 %0, %1
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  %call = call i32 @ave(i32 %add, i32 %add)
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32* %data, i64 %idxprom
  store i32 %call, i32* %arrayidx, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

!nvvm.annotations = !{!1}
!1 = metadata !{void (i32*)* @simple, metadata !"kernel", i32 1}
