=================================
CUDA Libraries Host API
=================================

NumbaPro provides binding to cuRAND, cuFFT host API for operation on numpy
arrays.

cuRAND
======

Provides pseudo-random number generator and quasi-random generator.

class PRNG
-----------

.. autoclass:: numbapro.cudalib.curand.PRNG
   :members:
   

class QRNG
------------

.. autoclass:: numbapro.cudalib.curand.QRNG
   :members:


Top Level PRNG Functions
--------------------------

Simple interface to the PRNG methods.  

.. note:: This methods automatically create a PRNG object.

.. autofunction:: numbapro.cudalib.curand.uniform

.. autofunction:: numbapro.cudalib.curand.normal

.. autofunction:: numbapro.cudalib.curand.lognormal

.. autofunction:: numbapro.cudalib.curand.poisson

Top Level QRNG Functions
--------------------------

Simple interface to the QRNG methods.  

.. note:: This methods automatically create a QRNG object.

.. autofunction:: numbapro.cudalib.curand.quasi


cuFFT
=======

Provides FFT and inverse FFT for 1D, 2D and 3D arrays.


.. note::  cuFFT only supports FFT operations on numpy.float32, numpy float64,
           numpy.complex32, numpy.complex64 with C-contiguous datalayout.


Forward FFT
------------

.. py:function:: numbapro.cudalib.cufft.fft(ary, out[, stream])
.. py:function:: numbapro.cudalib.cufft.fft_inplace(ary[, stream])

    :param ary: The input array. The inplace version stores the result in here.
    :param out: The output array for non-inplace versions.
    :param stream: The CUDA stream in which all operations will take place.


Inverse FFT
------------

.. py:function:: numbapro.cudalib.cufft.ifft(ary, out[, stream])
.. py:function:: numbapro.cudalib.cufft.ifft_inplace(ary[, stream])

    :param ary: The input array. The inplace version stores the result in here.
    :param out: The output array for non-inplace versions.
    :param stream: The CUDA stream in which all operations will take place.

FFTPlan
--------

.. autoclass:: numbapro.cudalib.cufft.FFTPlan
    :members:


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

