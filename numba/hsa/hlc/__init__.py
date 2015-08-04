from __future__ import absolute_import

import os

DATALAYOUT = {
    64: ("e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256"
         "-v512:512-v1024:1024-n32"),
}

TRIPLE = "hsail64-pc-unknown-amdopencl"

# Allow user to use "NUMBA_USE_LIBHLC" env-var to use cmdline HLC.
if os.environ.get('NUMBA_USE_LIBHLC', '').lower() not in ['0', 'no', 'false']:
    from . import libhlc as hlc
