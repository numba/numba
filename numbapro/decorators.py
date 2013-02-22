from numba import jit as numba_jit, autojit as numba_autojit
from numbapro.pipeline import numbapro_env

def autojit(*args, **kwds):
    kwds.setdefault('env', numbapro_env)
    return numba_autojit(*args, **kwds)

def jit(*args, **kwds):
    kwds.setdefault('env', numbapro_env)
    return numba_jit(*args, **kwds)
