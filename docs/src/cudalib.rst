=================================
CUDA Libraries Host API
=================================

NumbaPro provides binding to cuRAND, cuFFT host API for operation on numpy
arrays.

cuRAND
======

Provides pseudo-random number generator and quasi-random generator.

The functionalities are available under::

    from numbapro.cudalib import curand


uniform
-------

::

    curand.uniform(size, dtype=np.float, device=False)

Generate and return floating point random numbers sampled from a uniform
distribution.

- `size`: Size of the output array.
- `dtype`: Optional; default to `np.float`. Controls the data-type of the output
           array.  Can only be `np.float32` or `np.float64`
- `device`: Optional; default to `False`. Set to `True` to return a device
            array; otherwise, a numpy array is returned.

normal
-------

::

    curand.normal(mean, sigma, size, dtype=np.float, device=False)

Generate and return floating point random numbers sampled from a normal
distribution.

- `mean`: Center point of the distribution.
- `sigma`: Standard deviation of the distribution.
- `size`: Size of the output array.
- `dtype`: Optional; default to `np.float`. Controls the data-type of the output
           array.  Can only be `np.float32` or `np.float64`
- `device`: Optional; default to `False`. Set to `True` to return a device
            array; otherwise, a numpy array is returned.

lognormal
----------

::

    curand.lognormal(mean, sigma, size, dtype=np.float, device=False)


Generate and return floating point random numbers sampled from a log-normal
distribution.

- `mean`: Center point of the distribution.
- `sigma`: Standard deviation of the distribution.
- `size`: Size of the output array.
- `dtype`: Optional; default to `np.float`. Controls the data-type of the output
           array.  Can only be `np.float32` or `np.float64`
- `device`: Optional; default to `False`. Set to `True` to return a device
            array; otherwise, a numpy array is returned.

poisson
--------

::

    curand.poisson(lmbd, size, device=False)

Generate and return int32 random numbers sampled from a poisson distribution.

- `lmbd`: lambda
- `size`: Size of the output array.
- `device`: Optional; default to `False`. Set to `True` to return a device
            array; otherwise, a numpy array is returned.

quasi
------

::

    curand.quasi(size, bits=32, nd=1, device=False)

Generate and return quasi random number using SOBOL{bits} RNG type

- `size`: Size of the output array.
- `bits`: Bit length of output element; e.g. 32 or 64.
- `nd`: Number of dimension.
- `device`: Optional; default to `False`. Set to `True` to return a device
            array; otherwise, a numpy array is returned.

class PRNG
-----------


::

    curand.PRNG(rndtype=PRNG.DEFAULT, seed=None, offset=None, stream=None)

Pseudo random number generator.

- `rndtype`: Optional. Default to `PRNG.DEFAULT`.
             Possible value: `PRNG.TEST`, `PRNG.DEFAULT`, `PRNG.XORWOW`,
             `PRNG.MRG32K3A`, `PRNG.MTGP32`.
- `seed`: Optional. Default to system default. An integer to use as the
          seed for the PRNG.
- `offset`: Optional. Default to system default. The offset to the random
            number stream.
- `stream`: Optional. Default to system default. A CUDA stream. All
            operations by this PRNG will be placed on the given CUDA stream.

The PRNG class provides the following methods::

    PRNG.uniform(ary, size=None)
    PRNG.normal(ary, mean, sigma, size=None)
    PRNG.lognormal(ary, mean, sigma, size=None)
    PRNG.poisson(ary, lmbd, size=None)

These methods is similar to the top-level function shown in the previous
section.  The difference is that they writes the output to the `ary` argument.

- `ary`: A numpy or device array in which the random numbers will be written.
         The output is automtically transferred back to the host
         only if a numpy array is provided
- `size`: Optional. Default to the size of `ary`.
- `mean`: Center of the normal distribution.
- `sigma`: Standard deviation of the normal distribution.
- `lmbd`: Lambda for the poisson distribution.


class QRNG
-----------

Quasi random number generator.

- `rndtype`: Optional. Default to `QRNG.DEFAULT`. 
             Possible value: `QRNG.TEST`,
             `QRNG.DEFAULT`, `QRNG.SOBOL32`, `QRNG.SCRAMBLED_SOBOL32`, 
             `QRNG.SOBOL64`, `QRNG.SCRAMBLED_SOBOL64`.
- `seed`: Optional. Default to system default. An integer to use as
            the seed for the PRNG.
- `ndim`:  Optional.  Default to system default.  Number of dimension.
- `offset`: Optional. Default to system default. The offset to the random
            number stream.
- `stream`: Optional. Default to system default. A CUDA stream. All
            operations by this
            PRNG will be placed on the given CUDA stream.

The QRNG class provide one method::

    QRNG.generate(ary, size=None)

- `ary`: A numpy or device array in which the random numbers will be written.
         The output is automtically transferred back to the host
         only if a numpy array is provided
- `size`: Optional. Default to the size of `ary`.



cuFFT
=======

