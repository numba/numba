from __future__ import absolute_import, print_function
from numba.targets.registry import target_registry
from numba.npyufunc import Vectorize, GUVectorize


_is_initialize = False


def initialize_gpu_target():
    global _is_initialize
    _init_numba_jit_registry()
    _is_initialize = True


def ensure_cuda_support():
    if not _is_initialize:
        from numba.cuda import cuda_error

        raise cuda_error


def init_jit():
    from numbapro.cudapy.dispatcher import CUDADispatcher

    return CUDADispatcher


def init_vectorize():
    from numbapro.cudavec.vectorizers import CudaVectorize

    return CudaVectorize


def init_guvectorize():
    from numbapro.cudavec.vectorizers import CudaGUFuncVectorize

    return CudaGUFuncVectorize


def _init_numba_jit_registry():
    target_registry.ondemand['gpu'] = init_jit
    Vectorize.target_registry.ondemand['gpu'] = init_vectorize
    GUVectorize.target_registry.ondemand['gpu'] = init_guvectorize

