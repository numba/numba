#
# Public
#
last_error = None

_is_initialize = False

def initialize():
    "Safe to run multiple times"
    global _is_initialize
    if _is_initialize: return
    from .error import CudaSupportError, NvvmSupportError
    global last_error
    try:
        _init_driver()
        _init_nvvm()
        _init_numba_jit_registry()
        _is_initialize = True
        return True
    except CudaSupportError, e:
        last_error = e
        _init_poison_jit_registry()
        return False
    except NvvmSupportError, e:
        last_error = e
        _init_poison_jit_registry()
        return False
#
# Privates
#

def _init_driver():
    from .driver import Driver
    Driver() # raises CudaSupportError

def _init_nvvm():
    from .nvvm import NVVM
    NVVM() # raises NvvmSupportError

def _init_numba_jit_registry():
    from .decorators import cuda_jit, CudaAutoJitNumbaFunction
    from numba.decorators import jit_targets, autojit_wrappers
    jit_targets[('gpu', 'ast')] = cuda_jit
    autojit_wrappers[('gpu', 'ast')] = CudaAutoJitNumbaFunction

def _init_poison_jit_registry():
    from numba.decorators import jit_targets
    def poison(*args, **kws):
        raise last_error
    jit_targets[('gpu', 'ast')] = poison

