from _declaration import Declaration

uniform = Declaration('uniform', '''
uniform(ary)

    Write random floating-point values into `ary`.
    The values are sampled from a uniform distribution.

    ary --- a contiguous array of float or double.

''')

normal  = Declaration('normal', '''
normal(ary)

    Write random floating-point values into `ary`.
    The values are sampled from a normal distribution.

    ary --- a contiguous array of float or double.
''')

#
# Implementation
#
def _get_or_insert_prng(cu):
    from time import time
    from numbapro.cudalib import curand
    skey = 'prng'
    prng = cu._state.get(skey)
    if prng is None:
        prng = curand.PRNG(seed=int(time()), stream=cu._stream)
        cu._state.set(skey, prng)
    return prng


@uniform.register('gpu')
def gpu_uniform(cu, ntid, args):
    import numpy as np
    # check arguments
    (ary,) = args
    if ary.dtype != np.float32 and ary.dtype != np.float64:
        raise ValueError("Must be an array of float or double")
    # setup PRNG
    prng = _get_or_insert_prng(cu)
    prng.uniform(ary, ntid)
    

@normal.register('gpu')
def gpu_normal(cu, ntid, args):
    import numpy as np
    # check arguments
    (ary,) = args
    if ary.dtype != np.float32 and ary.dtype != np.float64:
        raise ValueError("Must be an array of float or double")
    # setup PRNG
    prng = _get_or_insert_prng(cu)
    prng.normal(ary, ntid)

