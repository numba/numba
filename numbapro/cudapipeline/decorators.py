import numba
from .environment import CudaEnvironment

def cuda_jit(restype=None, argtypes=None, nopython=False,
         _llvm_module=None, env_name=None, env=None, **kwargs):
    # error handling
    if restype != numba.void:
        raise TypeError("CUDA kernel must have void return type.")
    assert _llvm_module is None
    assert env_name is None
    assert env is None
    # real work
    nopython = True       # override nopython flag
    env = CudaEnvironment.get_environment('numbapro.cuda')
    def _jit_decorator(func):
        envsp = env.specializations
        envsp.register(func)
        result = envsp.compile_function(func, argtypes, restype=restype,
                                        nopython=nopython, ctypes=False,
                                        compile_only=True, **kwargs)
        sig, lfunc, pycallable = result
        assert pycallable is None
        print lfunc
#        print env.llvm_context.module
        raise NotImplementedError

    return _jit_decorator
