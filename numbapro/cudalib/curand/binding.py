import numpy as np
from ctypes import (c_float, c_int, c_void_p, POINTER, byref, cast, c_ulonglong,
                    c_uint, c_double, c_size_t)

from numbapro.cudadrv.drvapi import cu_stream
from numbapro.cudadrv.driver import device_pointer, host_pointer
from numbapro.cudalib.libutils import Lib, ctype_function
from numbapro._utils import finalizer

# enum curandStatus
STATUS = {
    0: ('CURAND_STATUS_SUCCESS',
        'No errors'),
    100: ('CURAND_STATUS_VERSION_MISMATCH',
          'Header file and linked library version do not match'),
    101: ('CURAND_STATUS_NOT_INITIALIZED',
          'Generator not initialized'),
    102: ('CURAND_STATUS_ALLOCATION_FAILED',
          'Memory allocation failed'),
    103: ('CURAND_STATUS_TYPE_ERROR',
          'Generator is wrong type'),
    104: ('CURAND_STATUS_OUT_OF_RANGE',
          'Argument out of range'),
    105: ('CURAND_STATUS_LENGTH_NOT_MULTIPLE',
          'Length requested is not a multple of dimension'),
    106: ('CURAND_STATUS_DOUBLE_PRECISION_REQUIRED',
          'GPU does not have double precision required by MRG32k3a'),
    201: ('CURAND_STATUS_LAUNCH_FAILURE',
          'Kernel launch failure'),
    202: ('CURAND_STATUS_PREEXISTING_FAILURE',
          'Preexisting failure on library entry'),
    203: ('CURAND_STATUS_INITIALIZATION_FAILED',
          'Initialization of CUDA failed'),
    204: ('CURAND_STATUS_ARCH_MISMATCH',
          'Architecture mismatch, GPU does not support requested feature'),
    999: ('CURAND_STATUS_INTERNAL_ERROR',
          'Internal library error'),
}
curandStatus_t = c_int


# enum curandRngType
CURAND_RNG_TEST = 0
## Default pseudorandom generator
CURAND_RNG_PSEUDO_DEFAULT = 100
## XORWOW pseudorandom generator
CURAND_RNG_PSEUDO_XORWOW = 101
## MRG32k3a pseudorandom generator
CURAND_RNG_PSEUDO_MRG32K3A = 121
## Mersenne Twister pseudorandom generator
CURAND_RNG_PSEUDO_MTGP32 = 141
## Default quasirandom generator
CURAND_RNG_QUASI_DEFAULT = 200
## Sobol32 quasirandom generator
CURAND_RNG_QUASI_SOBOL32 = 201
## Scrambled Sobol32 quasirandom generator
CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202
## Sobol64 quasirandom generator
CURAND_RNG_QUASI_SOBOL64 = 203
## Scrambled Sobol64 quasirandom generator
CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204
curandRngType_t = c_int

# enum curandOrdering
## Best ordering for pseudorandom results
CURAND_ORDERING_PSEUDO_BEST = 100
## Specific default 4096 thread sequence for pseudorandom results
CURAND_ORDERING_PSEUDO_DEFAULT = 101
## Specific seeding pattern for fast lower quality pseudorandom results
CURAND_ORDERING_PSEUDO_SEEDED = 102
## Specific n-dimensional ordering for quasirandom results
CURAND_ORDERING_QUASI_DEFAULT = 201
curandOrdering_t = c_int

# enum curandDirectionVectorSet
## Specific set of 32-bit direction vectors generated from polynomials
## recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101
## Specific set of 32-bit direction vectors generated from polynomials
## recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions,
## and scrambled
CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102
## Specific set of 64-bit direction vectors generated from polynomials
## recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103
## Specific set of 64-bit direction vectors generated from polynomials
## recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions,
## and scrambled
CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104
curandDirectionVectorSet_t = c_int

# enum curandMethod
CURAND_CHOOSE_BEST = 0
CURAND_ITR = 1
CURAND_KNUTH = 2
CURAND_HITR = 3
CURAND_M1 = 4
CURAND_M2 = 5
CURAND_BINARY_SEARCH = 6
CURAND_DISCRETE_GAUSS = 7
CURAND_REJECTION = 8
CURAND_DEVICE_API = 9
CURAND_FAST_REJECTION = 10
CURAND_3RD = 11
CURAND_DEFINITION = 12
CURAND_POISSON = 13
curandMethod_t = c_int

curandGenerator_t = c_void_p
p_curandGenerator_t = POINTER(curandGenerator_t)


class CuRandError(Exception):
    def __init__(self, code):
        super(CuRandError, self).__init__(STATUS[code])


class libcurand(Lib):
    lib = 'curand'
    ErrorType = CuRandError

    @property
    def version(self):
        ver = c_int(0)
        self.curandGetVersion(byref(ver))
        return ver.value

    curandGetVersion = ctype_function(curandStatus_t, POINTER(c_int))

    curandCreateGenerator = ctype_function(
        curandStatus_t,
        p_curandGenerator_t, # generator reference
        curandRngType_t)     # rng_type

    curandDestroyGenerator = ctype_function(
        curandStatus_t,
        curandGenerator_t)

    curandSetStream = ctype_function(curandStatus_t,
                                     cu_stream)

    curandSetGeneratorOffset = ctype_function(curandStatus_t,
                                              curandGenerator_t,
                                              c_ulonglong)

    curandSetPseudoRandomGeneratorSeed = ctype_function(
        curandStatus_t,
        curandGenerator_t,
        c_ulonglong)

    curandSetQuasiRandomGeneratorDimensions = ctype_function(
        curandStatus_t,
        curandGenerator_t,
        c_uint)

    curandGenerate = ctype_function(curandStatus_t,
                                    curandGenerator_t,
                                    POINTER(c_uint))

    curandGenerateLongLong = ctype_function(curandStatus_t,
                                            curandGenerator_t,
                                            POINTER(c_ulonglong))

    curandGenerateUniform = ctype_function(curandStatus_t,
                                           curandGenerator_t,
                                           POINTER(c_float),
                                           c_size_t)

    curandGenerateUniformDouble = ctype_function(curandStatus_t,
                                                 curandGenerator_t,
                                                 POINTER(c_double),
                                                 c_size_t)

    curandGenerateNormal = ctype_function(curandStatus_t,
                                          curandGenerator_t,
                                          POINTER(c_float),
                                          c_size_t,
                                          c_float,
                                          c_float)

    curandGenerateNormalDouble = ctype_function(curandStatus_t,
                                                curandGenerator_t,
                                                POINTER(c_double),
                                                c_size_t,
                                                c_double,
                                                c_double)

    curandGenerateLogNormal = ctype_function(curandStatus_t,
                                             curandGenerator_t,
                                             POINTER(c_float),
                                             c_size_t,
                                             c_float,
                                             c_float)

    curandGenerateLogNormalDouble = ctype_function(curandStatus_t,
                                                   curandGenerator_t,
                                                   POINTER(c_double),
                                                   c_size_t,
                                                   c_double,
                                                   c_double)

    curandGeneratePoisson = ctype_function(curandStatus_t,
                                           curandGenerator_t,
                                           POINTER(c_uint),
                                           c_size_t,
                                           c_double)


class Generator(finalizer.OwnerMixin):
    def __init__(self, rng_type=CURAND_RNG_TEST):
        self._api = libcurand()
        self._handle = curandGenerator_t(0)
        self._api.curandCreateGenerator(byref(self._handle), rng_type)
        self._finalizer_track((self._handle, self._api))

    @classmethod
    def _finalize(cls, res):
        handle, api = res
        api.curandDestroyGenerator(handle)

    def set_stream(self, stream):
        return self._api.curandSetStream(self._handle, stream.handle)

    def set_offset(self, offset):
        return self._api.curandSetGeneratorOffset(self._handle, offset)

    def set_pseudo_random_generator_seed(self, seed):
        return self._api.curandSetPseudoRandomGeneratorSeed(self._handle, seed)

    def set_quasi_random_generator_dimensions(self, num_dim):
        return self._api.curandSetQuasiRandomGeneratorDimensions(self._handle,
                                                                 num_dim)

    def generate(self, devout, num):
        fn, ptr = self.__uint32_or_uint64(devout,
                                          self._api.curandGenerate,
                                          self._api.curandGenerateLongLong)
        return fn(self._handle, ptr, num)

    def generate_uniform(self, devout, num):
        '''
        devout --- device array for the output
        num    --- # of float to generate
        '''
        fn, ptr = self.__float_or_double(devout,
                                         self._api.curandGenerateUniform,
                                         self._api.curandGenerateUniformDouble)
        return fn(self._handle, ptr, num)

    def generate_normal(self, devout, num, mean, stddev):
        fn, ptr = self.__float_or_double(devout,
                                         self._api.curandGenerateNormal,
                                         self._api.curandGenerateNormalDouble)
        return fn(self._handle, ptr, num, mean, stddev)

    def generate_log_normal(self, devout, num, mean, stddev):
        fn, ptr = self.__float_or_double(
            devout,
            self._api.curandGenerateLogNormal,
            self._api.curandGenerateLogNormalDouble)
        return fn(self._handle, ptr, num, mean, stddev)

    def generate_poisson(self, devout, num, lmbd):
        if devout.dtype not in (np.dtype(np.uint32), np.dtype(np.int32)):
            raise ValueError("Only accept int32 or uint32 arrays")
        dptr = device_pointer(devout)
        ptr = cast(c_void_p(dptr), POINTER(c_uint))
        return self._api.curandGeneratePoisson(self._handle, ptr, num, lmbd)

    def __float_or_double(self, devary, floatfn, doublefn):
        if devary.dtype == np.float32:
            fn = floatfn
            fty = c_float
        elif devary.dtype == np.float64:
            fn = doublefn
            fty = c_double
        else:
            raise ValueError("Only accept float or double arrays.")
        dptr = device_pointer(devary)
        ptr = cast(c_void_p(dptr), POINTER(fty))
        return fn, ptr

    def __uint32_or_uint64(self, devary, fn32, fn64):
        if devary.dtype in (np.dtype(np.uint32), np.dtype(np.int32)):
            fn = self._api.curandGenerate
            ity = c_uint
        elif devary.dtype in (np.dtype(np.uint64), np.dtype(np.int64)):
            fn = self._api.curandGenerateLongLong
            ity = c_ulonglong
        else:
            raise ValueError("Only accept int32, int64, "
                             "uint32 or uint64 arrays")
        dptr = device_pointer(devary)
        ptr = cast(c_void_p(dptr), POINTER(ity))
        return fn, ptr
