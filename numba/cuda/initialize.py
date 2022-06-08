def initialize_all():
    # Import models to register them with the data model manager
    import numba.cuda.models  # noqa: F401

    from numba import cuda
    from numba.cuda.dispatcher import CUDADispatcher
    from numba.core.target_extension import (target_registry,
                                             dispatcher_registry,
                                             jit_registry)

    def cuda_jit_device(*args, **kwargs):
        kwargs['device'] = True
        return cuda.jit(*args, **kwargs)

    cuda_target = target_registry["cuda"]
    jit_registry[cuda_target] = cuda_jit_device
    dispatcher_registry[cuda_target] = CUDADispatcher
