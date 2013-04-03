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

Provides FFT and inverse FFT for 1D, 2D and 3D arrays.

All functionalities are provided under::

    from numbapro.cudalib import cufft
    
    
Supported types and operations
-------------------------------

cuFFT only supports FFT operations on numpy.float32, numpy float64, 
numpy.complex32, numpy.complex64 with C-contiguous datalayout.


fft, fft_inplace, ifft, ifft_inplace
----------------------------------------

::

    fft(ary, out, stream=None)
    fft_inplace(ary, stream=None)
    ifft(ary, out, stream=None)
    ifft_inplace(ary, stream=None)
    
The `fft` and `fft_inplace` functions compute the forward FFT.
The `ifft` and `ifft_inplace` functions compute the inverse FFT.
The output is stored in `out` or in `ary`
with the inplace version is used.  Both `ary` and `out` can be numpy array
or device array.

- `ary`: The input array. The inplace version stores the result in here.
- `out`: The output array.
- `stream`: The CUDA stream in which all operations will take place.

class FFTPlan
---------------

Represents a cuFFT Plan.

:: 

    FFTPlan(shape, itype, otype, batch=1, stream=0, mode=FFTPLAN.MODE_DEFAULT)
    
Instantiate a new FFTPlan object.

- `shape`: Input array shape.
- `itype`: Input array dtype.
- `otype`: Output array dtype.
- `batch`: Optional. Maximum number of operation the plan can perform.
- `stream`: Optional. A CUDA stream for all the operations to put on.
- `mode`: Optional. cuFFT compatability mode.  Default to use FFTW padding
          datalayout.  Other possible values are: `FFTPLAN.MODE_NATIVE`,
          `FFTPLAN.MODE_FFTW_PADDING`, `FFTPLAN.MODE_FFTW_ASYMMETRIC`,
          `FFTPLAN.MODE_FFTW_ALL`, `FFTPLAN.MODE_FFTW_DEFAULT`.
          
The class provides two methods::
    
        forward(ary, out=None)
        inverse(ary, out=None)
            
The `forward` method computes the forward FFT on ary and stores the result to
`out`.  If `out` is None, an inplace operation is performed.

The `inverse` method computes the inverse FFT on ary and stores the result to
`out`.  If `out` is None, an inplace operation is performed.


cuBLAS
=======

Provides basic linear algebra building blocks that operates on the CUDA device.

All functionalities are provided under::
        
    from numbapro.cudalib import cublas

class Blas
-----------

Currently only level-1 BLAS routines are implemented.

::

    blas = Blas(stream=0)

Instantiate a new Blas object.

- `stream`: Optional. A CUDA stream for all the operations to put on.


The following methods under the `Blas` object corresponds to the level-1 cuBLAS
routines.  These method dispatches to different cuBLAS routines depending on the
input type.  All array inputs can be either host or device arrays.

Blas.nrm2
----------

::
    
    l2norm = blas.nrm2(x)
    
Computes the L2 norm for array `x`.  Same as `np.linalg.norm(x)`.


Blas.dot
----------

::
    
    dotprod = blas.dot(x, y)
    
Compute the dot product of array `x` and array `y`.  Same as `np.dot(x, y)`.


Blas.dotu
----------

Compute the dot product of array `x` and array `y` for complex dtype only.
Same as `np.dot(x, y)`.

Blas.dotc
----------

Uses the conjugate of the element of the vectors to compute the dot product of
array `x` and array `y` for complex dtype only.  Same as `np.vdot(x, y)`.

Blas.scal
----------

::

    blas.scal(alpha, x)

Scale `x` inplace by alpha.  Same as `x = alpha * x`.

Blas.axpy
----------

::

    blas.axpy(alpha, x)

Compute `y = alpha * x + y` inplace.


Blas.amax
----------

::

    maxidx = blas.amax(x)
    
Find the index of the first largest element in array `x`.  Same as `np.argmax(x)`


Blas.amin
----------

::

    minidx = blas.amin(x)
    
Find the index of the first largest element in array `x`.  Same as `np.argmin(x)`

Blas.asum
----------

::

    sum = blas.sum(x)
    
Compute the sum of all element in array `x`.


Blas.rot
----------

::

    blas.rot(x, y, c, s)
    
Apply the Givens rotation matrix specified by the cosine element `c` and the 
sine element `s` inplace on vector element `x` and `y`.

Same as `x, y = c * x + s * y, -s * x + c * y`


Blas.rotg
----------

::

    blas.rotg(a, b)
    
Constructs the Givens rotation matrix with the column vector (a, b).
Returns r, z, c, s.

r --- `r = a**2 + b**2`
z --- Use to reconstruct `c` and `s`.  Refer to cuBLAS documentation for detail.
c --- The consine element.
s --- The sine element.

Blas.rotm
----------

::

    blas.rotm(x, y, param)
    
Applies the modified Givens transformation inplace.

Same as::

    param = flag, h11, h21, h12, h22
    x[i] = h11 * x[i] + h12 * y[i]
    y[i] = h21 * x[i] + h22 * y[i]

Refer to the cuBLAS documentation for the use of `flag`.

Blas.rotmg
----------

::
    
    param = blas.rotmg(d1, d2, x1, y1)
    
Constructs the modified Givens transformation `H` that zeros out the second
entry of a column vector `(d1 * x1, d2 * y1)`.

Returns a 1D array that is usable in `rotm`.  The first element is the flag
for `rotm`.  The rest of the elements corresponds to the `h11, h21, h12, h22`
elements of `H`.

