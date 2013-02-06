# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# THIS IS A BRANCH-SPECIFIC MODULE INTENDED TO ULTIMATELY BE A DROP-IN
# REPLACEMENT FOR THE NUMBA DECORATOR MODULE!!!
#
# If you spot this module outside the pipelineenv branch, please contact
# your nearest Numba developer immediately!
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

__all__ = 'autojit', 'jit', 'export', 'exportmany'

from numba import environment

def autojit(*args, **kws):
    env = environment.NumbaEnvironment.get_environment(kws.pop('env', None))
    raise NotImplementedError('XXX')

def jit(*args, **kws):
    env = environment.NumbaEnvironment.get_environment(kws.pop('env', None))
    raise NotImplementedError('XXX')

def export(*args, **kws):
    env = environment.NumbaEnvironment.get_environment(kws.pop('env', None))
    raise NotImplementedError('XXX')

def exportmany(*args, **kws):
    env = environment.NumbaEnvironment.get_environment(kws.pop('env', None))
    raise NotImplementedError('XXX')
