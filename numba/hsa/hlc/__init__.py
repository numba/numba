from __future__ import absolute_import

import os

DATALAYOUT = {
    # 32: ("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32"
    # ":32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64"
    # ":64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512"
    # ":512:512-v1024:1024:1024"),
    # 64: ("e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:"
    #      "512-v1024:1024-n32"),
    64: ("e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256"
         "-v512:512-v1024:1024-n32"),
}

TRIPLE = "hsail64-pc-unknown-amdopencl"

if os.environ.get('NUMBA_USE_LIBHLC', '').lower() not in ['0', 'no', 'false']:
    from . import libhlc as hlc
