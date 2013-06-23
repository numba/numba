from numba import jit as numba_jit, autojit as numba_autojit
from numbapro.pipeline import numbapro_env
from numbapro.cudadrv.initialize import last_error
from numbapro import cuda

def autojit(*args, **kwds):
    target = kwds.get('target')
    if target == 'gpu':
        if last_error is not None:
            raise last_error
        return cuda.autojit
    else:
        kwds.setdefault('env', numbapro_env)
        return numba_autojit(*args, **kwds)

def jit(*args, **kwds):
    target = kwds.get('target')
    if target == 'gpu':
        if last_error is not None:
            raise last_error
        return cuda.jit
    else:
        kwds.setdefault('env', numbapro_env)
        return numba_jit(*args, **kwds)
