from __future__ import absolute_import, print_function


def init_jit():
    from numba.cuda.dispatcher import CUDADispatcher
    return CUDADispatcher

def initialize_all():
    from numba.targets.registry import dispatcher_registry
    dispatcher_registry.ondemand['gpu'] = init_jit
    dispatcher_registry.ondemand['cuda'] = init_jit
