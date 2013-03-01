from numba import jit as numba_jit, autojit as numba_autojit
from numbapro.pipeline import numbapro_env

def autojit(*args, **kwds):
    target = kwds.get('target')
    if target == 'gpu':
        from numbapro.cudapipeline.environment import CudaEnvironment
        from numbapro.cudapipeline.initialize import last_error
        if last_error is not None:
            raise last_error
        kwds.setdefault('env', CudaEnvironment.get_environment('numbapro.cuda'))
    else:
        kwds.setdefault('env', numbapro_env)
    return numba_autojit(*args, **kwds)

def jit(*args, **kwds):
    target = kwds.get('target')
    if target == 'gpu':
        from numbapro.cudapipeline.environment import CudaEnvironment
        from numbapro.cudapipeline.initialize import last_error
        if last_error is not None:
            raise last_error
        kwds.setdefault('env', CudaEnvironment.get_environment('numbapro.cuda'))
    else:
        kwds.setdefault('env', numbapro_env)
    return numba_jit(*args, **kwds)
