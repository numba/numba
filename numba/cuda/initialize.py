from __future__ import absolute_import, print_function
from numba.targets.registry import target_registry

def init_jit():
    from numba.cuda.dispatcher import CUDADispatcher
    return CUDADispatcher

def initialize_all():
    target_registry.ondemand['gpu'] = init_jit
    target_registry.ondemand['cuda'] = init_jit
