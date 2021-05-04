def initialize_all():
    # Import models to register them with the data model manager
    import numba.cuda.models  # noqa: F401

    from numba import cuda
    from numba.cuda.compiler import Dispatcher
    from numba.core import decorators
    from numba.core.extending_hardware import (hardware_registry,
                                               dispatcher_registry)

    def cuda_jit_device(*args, **kwargs):
        kwargs['device'] = True
        return cuda.jit(*args, **kwargs)

    cuda_hw = hardware_registry["cuda"]
    decorators.jit_registry[cuda_hw] = cuda_jit_device
    dispatcher_registry[cuda_hw] = Dispatcher
