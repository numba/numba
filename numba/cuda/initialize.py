from __future__ import absolute_import, print_function
from numba.targets.registry import target_registry
from numba.npyufunc import Vectorize, GUVectorize


def init_jit():
    from numba.cuda.dispatcher import CUDADispatcher
    return CUDADispatcher


def init_vectorize():
    from numbapro.cudavec.vectorizers import CudaVectorize
    return CudaVectorize


def init_guvectorize():
    from numbapro.cudavec.vectorizers import CudaGUFuncVectorize
    return CudaGUFuncVectorize


def initialize_all():
    target_registry.ondemand['gpu'] = init_jit
    target_registry.ondemand['cuda'] = init_jit
    Vectorize.target_registry.ondemand['gpu'] = init_vectorize
    Vectorize.target_registry.ondemand['cuda'] = init_vectorize
    GUVectorize.target_registry.ondemand['gpu'] = init_guvectorize
    GUVectorize.target_registry.ondemand['cuda'] = init_guvectorize
