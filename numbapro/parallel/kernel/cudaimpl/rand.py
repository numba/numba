import numpy as np
from numbapro.parallel.kernel.builtins.random import seed, uniform, normal
from numbapro.cudalib import curand

def _get_or_insert_prng(cu, seed=None):
    skey = 'prng'
    prng = cu._state.get(skey)
    if prng is None:
        kws = {'stream': cu._stream}
        if seed is not None:
            kws['seed'] = int(seed)
        prng = curand.PRNG(**kws)
        cu._state.set(skey, prng)
    return prng

@seed.register('gpu')
def gpu_seed(cu, seed):
    _get_or_insert_prng(cu, seed=seed)

@uniform.register('gpu')
def gpu_uniform(cu, ntid, args):
    # check arguments
    (ary,) = args
    if ary.dtype != np.float32 and ary.dtype != np.float64:
        raise ValueError("Must be an array of float or double")
    # setup PRNG
    prng = _get_or_insert_prng(cu)
    prng.uniform(ary, ntid)
    

@normal.register('gpu')
def gpu_normal(cu, ntid, args):
    # check arguments
    mean = 0
    sigma = 1
    if len(args) == 3:
        (ary, mean, sigma) = args
    elif len(args) == 2:
        (ary, mean) = args
    else:
        (ary,) = args
    if ary.dtype != np.float32 and ary.dtype != np.float64:
        raise ValueError("Must be an array of float or double")
    # setup PRNG
    prng = _get_or_insert_prng(cu)
    prng.normal(ary, mean, sigma, ntid)

