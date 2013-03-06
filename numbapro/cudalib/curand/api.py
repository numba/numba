from . import binding
from numbapro.cuda import _auto_device

class RNG(object):
    "cuRAND pseudo random number generator"
    def __init__(self, gen):
        self._gen = gen
        self.__stream = 0

    @property
    def offset(self):
        return self.__offset

    @offset.setter
    def offset(self, offset):
        self.__offset = offset
        self._gen.set_offset(offset)

    @property
    def stream(self):
        '''Associate a CUDA stream to the generator object.
        All subsequent calls will use this stream.'''
        return self.__stream

    @stream.setter
    def stream(self, stream):
        self.__stream = stream
        self._gen.set_stream(stream)

    def _require_array(self, ary):
        if ary.ndim != 1:
            raise TypeError("Only accept 1-D array")
        if ary.strides[0] != ary.dtype.itemsize:
            raise TypeError("Only accept unit strided array")


class PRNG(RNG):

    TEST     = binding.CURAND_RNG_TEST
    DEFAULT  = binding.CURAND_RNG_PSEUDO_DEFAULT
    XORWOW   = binding.CURAND_RNG_PSEUDO_XORWOW
    MRG32K3A = binding.CURAND_RNG_PSEUDO_MRG32K3A
    MTGP32   = binding.CURAND_RNG_PSEUDO_MTGP32

    def __init__(self, rndtype=DEFAULT, seed=None, offset=None, stream=None):
        super(PRNG, self).__init__(binding.Generator(rndtype))
        self.rndtype = rndtype
        if seed is not None:
            self.seed = seed
        if offset is not None:
            self.offset = offset
        if stream is not None:
            self.stream = stream

    @property
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, seed):
        self.__seed = seed
        self._gen.set_pseudo_random_generator_seed(seed)

    def uniform(self, ary, size=None):
        self._require_array(ary)
        size = size or ary.size
        dary, conv = _auto_device(ary, stream=self.stream)
        self._gen.generate_uniform(dary, size)
        if conv:
            dary.to_host(stream=self.stream)

    def normal(self, ary, mean, sigma, size=None):
        self._require_array(ary)
        size = size or ary.size
        dary, conv = _auto_device(ary, stream=self.stream)
        self._gen.generate_normal(dary, size, mean, sigma)
        if conv:
            dary.to_host(stream=self.stream)


    def lognormal(self, ary, mean, sigma, size=None):
        self._require_array(ary)
        size = size or ary.size
        dary, conv = _auto_device(ary, stream=self.stream)
        self._gen.generate_log_normal(dary, size, mean, sigma)
        if conv:
            dary.to_host(stream=self.stream)


class QRNG(RNG):
    "cuRAND quasi random number generator"
    TEST                = binding.CURAND_RNG_TEST
    DEFAULT             = binding.CURAND_RNG_QUASI_DEFAULT
    SOBOL32             = binding.CURAND_RNG_QUASI_SOBOL32
    SCRAMBLED_SOBOL32   = binding.CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
    SOBOL64             = binding.CURAND_RNG_QUASI_SOBOL64
    SCRAMBLED_SOBOL64   = binding.CURAND_RNG_QUASI_SCRAMBLED_SOBOL64

    def __init__(self, rndtype=DEFAULT, ndim=None, offset=None, stream=None):
        super(QRNG, self).__init__(binding.Generator(rndtype))
        self.rndtype = rndtype
        if ndim is not None:
            self.ndim = ndim
        if offset is not None:
            self.offset = offset
        if stream is not None:
            self.stream = stream

    @property
    def ndim(self, ndim):
        return self.__ndim

    @ndim.setter
    def ndim(self, ndim):
        self.__ndim = ndim
        self._gen.set_quasi_random_generator_dimensions(ndim)

    def generate(self, ary, size=None):
        self._require_array(ary)
        size = size or ary.size
        dary, conv = _auto_device(ary, stream=self.stream)
        self._gen.generate(dary, size)
        if conv:
            dary.to_host(stream=self.stream)


