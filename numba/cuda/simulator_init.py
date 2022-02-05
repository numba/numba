# We import * from simulator here because * is imported from simulator_init by
# numba.cuda.__init__.
from .simulator import *
from .stubs import (threadIdx, blockIdx, blockDim, gridDim, laneid,
                    warpsize, syncthreads, syncthreads_count, syncwarp,
                    syncthreads_and, syncthreads_or, shared, local,
                    const, grid, gridsize, atomic, shfl_sync_intrinsic,
                    vote_sync_intrinsic, match_any_sync, match_all_sync,
                    threadfence_block, threadfence_system,
                    threadfence, selp, popc, brev, clz, ffs, fma, cbrt,
                    cg, activemask, lanemask_lt, nanosleep)


def is_available():
    """Returns a boolean to indicate the availability of a CUDA GPU.
    """
    # Simulator is always available
    return True


def cuda_error():
    """Returns None or an exception if the CUDA driver fails to initialize.
    """
    # Simulator never fails to initialize
    return None
