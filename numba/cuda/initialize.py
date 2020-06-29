def init_jit():
    from numba.cuda.dispatcher import CUDADispatcher
    return CUDADispatcher

def initialize_all():
    # Import models to register them with the data model manager
    import numba.cuda.models

    from numba.core.registry import dispatcher_registry
    dispatcher_registry.ondemand['gpu'] = init_jit
    dispatcher_registry.ondemand['cuda'] = init_jit
