import numpy
import numba
from numba import numbawrapper, jit as _numba_jit, autojit as _numba_autojit
from numba.decorators import compile_function
from .environment import CUEnvironment

def cu_jit(restype=None, argtypes=None, nopython=False,
           _llvm_module=None, env_name=None, env=None,
           **kwargs):
    # error handling
    if restype and restype != numba.void:
        raise TypeError("CU kernel must have void return type.")
    assert _llvm_module is None
    # real work
    nopython = True       # override nopython flag
    env_name = env_name or 'numbapro.cu'
    env = env or CUEnvironment.get_environment(env_name)
    def _jit_decorator(func):
        envsp = env.specializations
        envsp.register(func)
        result = compile_function(env, func, argtypes, restype=restype,
                                  nopython=nopython, ctypes=False,
                                  compile_only=True, **kwargs)
        sig, lfunc, pycallable = result
        assert pycallable is None
        return lfunc
    return _jit_decorator


