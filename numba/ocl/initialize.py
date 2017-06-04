from __future__ import absolute_import, print_function


def init_jit():
    from numba.ocl.dispatcher import OCLDispatcher
    return OCLDispatcher

def initialize_all():
    from numba.targets.registry import dispatcher_registry
    dispatcher_registry.ondemand['gpu'] = init_jit
    dispatcher_registry.ondemand['ocl'] = init_jit
