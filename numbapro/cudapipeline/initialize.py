#
# Public
#
def initialize():
    _init_numba_jit_registry()
    return True

#
# Privates
#

def _init_numba_jit_registry():
    from .decorators import cuda_jit
    from numba.decorators import jit_targets
    jit_targets[('gpu', 'ast')] = cuda_jit

