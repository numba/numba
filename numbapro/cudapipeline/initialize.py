#
# Public
#
last_error = None

def initialize():
    "Safe to run multiple times"
    from .error import CudaSupportError, NvvmSupportError
    global last_error
    try:
        _init_driver()
        _init_nvvm()
        _init_numba_jit_registry()

        # replace this function with a stub after the first run
        initialize = lambda: True
    #    globals()['initialize'] = lambda: True
        return True
    except CudaSupportError, e:
        last_error = e
        return False
    except NvvmSupportError, e:
        last_error = e
        return False
#
# Privates
#

def _init_driver():
    from .driver import Driver
    driver = Driver() # raises CudaSupportError

def _init_nvvm():
    from .nvvm import NVVM
    nvvm = NVVM() # raises NvvmSupportError

def _init_numba_jit_registry():
    from .decorators import cuda_jit, CudaAutoJitNumbaFunction
    from numba.decorators import jit_targets, autojit_wrappers
    jit_targets[('gpu', 'ast')] = cuda_jit
    autojit_wrappers[('gpu', 'ast')] = CudaAutoJitNumbaFunction

