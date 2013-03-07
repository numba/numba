from collections import namedtuple

impl_record = namedtuple('impl_record', ['name', 'impl'])

class Declaration(object):
    def __init__(self):
        self.__impls = {}

    def register(self, target, name=None):
        def _register(impl):
            self.__impls[target] = impl_record(name, impl)
            return impl
        return _register

    def get_implementation(self, target):
        return self.__impls[target]

rand = Declaration()

@rand.register('gpu')
def gpu_rand(cu, storage, ntid, args):
    from numbapro.cudalib import curand
    import numpy as np
    import time
    # check arguments
    (ary,) = args
    if ary.dtype != np.float32 and ary.dtype != np.float64:
        raise ValueError("Must be an array of integers of 32 or 64 bit length")
    # setup QRNG
    skey = 'prng'
    qrng = storage.get(skey)
    if qrng is None:
        qrng = curand.PRNG(seed=int(time.time()), stream=cu._stream)
        storage.set(skey, qrng)
    # call
    qrng.uniform(ary, ntid)
    
