import sys
import os

# where ROCM bitcode is installed
DEFAULT_ROCM_BC_PATH = '/opt/rocm/opencl/lib/x86_64/bitcode/'
ROCM_BC_PATH = os.environ.get("NUMBA_ROCM_BC_PATH", DEFAULT_ROCM_BC_PATH)
