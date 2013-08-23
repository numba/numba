from numba import jit as numba_jit, autojit as numba_autojit
from numbapro.cudadrv.initialize import last_error
from numbapro import cuda

def autojit(*args, **kwds):
    target = kwds.get('target')
    if target == 'gpu':
        if last_error is not None:
            raise last_error
        return cuda.autojit(*args, **kwds)
    else:
        return numba_autojit(*args, **kwds)

def jit(*args, **kwds):
    target = kwds.get('target')
    if target == 'gpu':
        if last_error is not None:
            raise last_error
        return cuda.jit(*args, **kwds)
    else:
        return numba_jit(*args, **kwds)
