def initialize_all():
    # Import models to register them with the data model manager
    import numba.cuda.models  # noqa: F401

    from numba import cuda
    from numba.core import decorators
    from numba.core.extending_hardware import hardware_registry

    def cuda_jit_device(*args, **kwargs):
        kwargs['device'] = True
        return cuda.jit(*args, **kwargs)

    decorators.jit_registry[hardware_registry["cuda"]] = cuda_jit_device
