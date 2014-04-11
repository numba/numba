from __future__ import absolute_import
from numba.cuda import is_available, cuda_error

if not is_available:
    raise cuda_error

from numba.cuda.cudadrv import devicearray, devices, driver
from .cudadrv import autotune


# Re export
from .cudapy.ptx import (threadIdx, blockIdx, blockDim, gridDim, syncthreads,
                         shared, local, const, grid, atomic)
from .cudapy import jit, autojit, declare_device

from numba.cuda import (
    require_context, current_context,
    to_device, device_array, pinned_array, mapped_array,
    synchronize, device_array_like, stream, pinned, mapped, event,
    select_device, get_current_device, list_devices, close,
    detect,
    )

from numba.cuda.api import _auto_device


def calc_occupancy(cc, reg, smem=0, smem_config=None):
    """Occupancy calculator

    Args
    ----
    - cc
        compute capability as a tuple-2 of ints.
    - reg
        register used per thread.
    - smem
        shared memory used per block.
    - smem_config
        (optional) smem configuration

    Returns
    -------
    returns an AutoTuner object

    """
    usage = {}
    usage['reg'] = reg
    usage['shared'] = smem
    at = autotune.AutoTuner(cc=cc, usage=usage, smem_config=smem_config)
    return at

