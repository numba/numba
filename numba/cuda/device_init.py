from __future__ import print_function, absolute_import, division

# Re export
from .stubs import (threadIdx, blockIdx, blockDim, gridDim, laneid,
                    warpsize, syncthreads, syncthreads_count, syncwarp,
                    syncthreads_and, syncthreads_or, shared, local,
                    const, grid, gridsize, atomic, shfl_sync_intrinsic,
                    vote_sync_intrinsic, match_any_sync, match_all_sync,
                    threadfence_block, threadfence_system,
                    threadfence, selp, popc, brev, clz, ffs, fma)
from .cudadrv.error import CudaSupportError
from .cudadrv import nvvm
from . import initialize
from .errors import KernelRuntimeError

from .decorators import jit, autojit, declare_device
from .api import *
from .api import _auto_device

from .kernels import reduction
reduce = Reduce = reduction.Reduce

from .intrinsic_wrapper import (all_sync, any_sync, eq_sync, ballot_sync,
                                shfl_sync, shfl_up_sync, shfl_down_sync,
                                shfl_xor_sync)


def is_available():
    """Returns a boolean to indicate the availability of a CUDA GPU.

    This will initialize the driver if it hasn't been initialized.
    """
    # whilst `driver.is_available` will init the driver itself,
    # the driver initialization may raise and as a result break
    # test discovery/orchestration as `cuda.is_available` is often
    # used as a guard for whether to run a CUDA test, the try/except
    # below is to handle this case.
    driver_is_available = False
    try:
        driver_is_available = driver.driver.is_available
    except CudaSupportError:
        pass

    return driver_is_available and nvvm.is_available()

def cuda_error():
    """Returns None or an exception if the CUDA driver fails to initialize.
    """
    return driver.driver.initialization_error

initialize.initialize_all()
