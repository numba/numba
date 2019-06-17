from __future__ import absolute_import

import os

# 32-bit private, local, and region pointers. 64-bit global, constant and flat.
# See:
# https://github.com/RadeonOpenCompute/llvm/blob/b20b796f65ab6ac12fac4ea32e1d89e1861dee6a/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp#L270-L275
# Alloc goes into addrspace(5) (private)
DATALAYOUT = {
    64: ("e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
         "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
         "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"),
}

TRIPLE = "amdgcn--amdhsa"

# Allow user to use "NUMBA_USE_LIBHLC" env-var to use cmdline HLC.
if os.environ.get('NUMBA_USE_LIBHLC', '').lower() not in ['0', 'no', 'false']:
    from . import libhlc as hlc
