=================================
CUDA Libraries Host API
=================================

NumbaPro provides binding to cuRAND, cuFFT host API for operation on numpy
arrays.

cuRAND
======

Provides pseudo-random number generator and quasi-random generator.
See `NVIDIA cuRAND <http://docs.nvidia.com/cuda/cuRAND/index.html>`_.

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
See `NVIDIA cuFFT <http://docs.nvidia.com/cuda/cufft/index.html>`_.


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
See `NVIDIA cuBLAS <http://docs.nvidia.com/cuda/cublas/index.html>`_.

The cuBlas binding provides a simpler interface to use numpy arrays and device
arrays.  We don't need special naming convention to identify the array types.
Type information are inferred from the given arguments.
Arguments for array storage information in cuBLAS C-API are not
necessary since numpy arrays and device arrays already contain the information.
Whenever an array is required in an argument, user can pass in numpy arrays
or device arrays.  The binding will automatically transfer any numpy arrays
to the device as needed.

.. autoclass:: numbapro.cudalib.cublas.Blas

BLAS Level 1
-------------

.. py:method:: numbapro.cudalib.cublas.Blas.norm2(x)

    Computes the L2 norm for array `x`. Same as `numpy.linalg.norm(x)`.

    :param x: input vector
    :type x: array
    :returns: resulting norm.

.. py:method:: numbapro.cudalib.cublas.Blas.dot(x, y)

    Compute the dot product of array `x` and array `y`.  Same as `np.dot(x, y)`.

    :param x: vector
    :type x: array
    :param y: vector
    :type y: array
    :returns: dot product of `x` and `y`

.. py:method:: numbapro.cudalib.cublas.Blas.dotu(x, y)

    Compute the dot product of array `x` and array `y` for complex dtype only.
    Same as `np.dot(x, y)`.

    :param x: vector
    :type x: array
    :param y: vector
    :type y: array
    :returns: dot product of `x` and `y`

.. py:method:: numbapro.cudalib.cublas.Blas.dotc(x, y)

    Uses the conjugate of the element of the vectors to compute the dot product
    of array `x` and array `y` for complex dtype only.  Same as `np.vdot(x, y)`.

    :param x: vector
    :type x: array
    :param y: vector
    :type y: array
    :returns: dot product of `x` and `y`


.. py:method:: numbapro.cudalib.cublas.Blas.scal(alpha, x)

    Scale `x` inplace by alpha.  Same as `x = alpha * x`

    :param alpha: scalar
    :param x: vector
    :type x: array

.. py:method:: numbapro.cudalib.cublas.Blas.axpy(alpha, x)

    Compute `y = alpha * x + y` inplace.

    :param alpha: scalar
    :param x: vector
    :type x: array


.. py:method:: numbapro.cudalib.cublas.Blas.amax(x)


    Find the index of the first largest element in array `x`.
    Same as `np.argmax(x)`

    :param x: vector
    :type x: array
    :returns: index (start from 0).


.. py:method:: numbapro.cudalib.cublas.Blas.amin(x)

    Find the index of the first largest element in array `x`.
    Same as `np.argmin(x)`

    :param x: vector
    :type x: array
    :returns: index (start from 0).


.. py:method:: numbapro.cudalib.cublas.Blas.asum(x)

    Compute the sum of all element in array `x`.

    :param x: vector
    :type x: array
    :returns: `x.sum()`

.. py:method:: numbapro.cudalib.cublas.Blas.rot(x, y, c, s)

    Apply the Givens rotation matrix specified by the cosine element `c` and the
    sine element `s` inplace on vector element `x` and `y`.

    Same as `x, y = c * x + s * y, -s * x + c * y`

    :param x: vector
    :type x: array
    :param y: vector
    :type y: array


.. py:method:: numbapro.cudalib.cublas.Blas.rotg(a, b)

    Constructs the Givens rotation matrix with the column vector (a, b).

    :param a: first element of the column vector
    :type a: scalar
    :param b: second element of the column vector
    :type b: scalar
    :returns: a tuple (r, z, c, s)

        r -- `r = a**2 + b**2`

        z -- Use to reconstruct `c` and `s`.
             Refer to cuBLAS documentation for detail.

        c -- The consine element.

        s -- The sine element.


.. py:method:: numbapro.cudalib.cublas.Blas.rotm(x, y, param)

    Applies the modified Givens transformation inplace.

    Same as::

        param = flag, h11, h21, h12, h22
        x[i] = h11 * x[i] + h12 * y[i]
        y[i] = h21 * x[i] + h22 * y[i]

    Refer to the cuBLAS documentation for the use of `flag`.

    :param x: vector
    :type x: array
    :param y: vector
    :type y: array


.. py:method:: numbapro.cudalib.cublas.Blas.rotmg(d1, d2, x1, y1)

    Constructs the modified Givens transformation `H` that zeros out the second
    entry of a column vector `(d1 * x1, d2 * y1)`.

    :param d1:
    :type d1: scalar
    :param d2:
    :type d2: scalar
    :param x1:
    :type x1: scalar
    :param y1:
    :type y1: scalar

    :returns: A 1D array that is usable in `rotm`.
              The first element is the flag for `rotm`.
              The rest of the elements corresponds to the `h11, h21, h12, h22`
              elements of `H`.

BLAS Level 2
-------------

All level 2 routines follow the following naming convention for all arguments:

* A, B, C, AP -- (2D array) Matrix argument.
                 `AP` implies packed storage for banded matrix.
* x, y, z -- (1D arrays)  Vector argument.
* alpha, beta -- (scalar) Can be floats or complex numbers depending.
* m -- (scalar)  Number of rows of matrix `A`.
* n -- (scalar)  Number of columns of matrix `A`.  If `m` is not needed,
                 `n` also means the number of rows of the matrix `A`; thus,
                 implying a square matrix.
* trans, transa, transb -- (string)
                'N' means the non-transpose operation is selected;
                'T' means the transpose operation is selected;
                'C' means the conjugate transpose operation is selected.
                `trans` only applies to the only matrix argument.
                `transa` and `transb` apply to matrix `A` and matrix `B`,
                respectively.
* uplo -- (string) Can be 'U' for filling the upper trianglar matrix; or 'L' for
          filling the lower trianglar matrix.
* diag -- (boolean)  Whether the matrix diagonal has unit elements.
* trans -- CUBLAS_OP_MAP
* mode -- (string) 'L' means the matrix is on the left side in the equation.
                   'R' means the matrix is on the right side in the equation.

.. note:: The last array argument is always overwritten with the result.

.. py:method:: numbapro.cudalib.cublas.Blas.gbmv(trans, m, n, kl, ku, alpha, A, x, beta, y')

    banded matrix-vector multiplication `y = alpha * op(A) * x + beta * y`

.. py:method:: numbapro.cudalib.cublas.Blas.gemv(trans, m, n, alpha, A, x, beta, y)

    matrix-vector multiplication `y = alpha * op(A) * x + beta * y`

.. py:method:: numbapro.cudalib.cublas.Blas.trmv(uplo, trans, diag, n, A, x)

    triangular matrix-vector multiplication `x = op(A) * x`

.. py:method:: numbapro.cudalib.cublas.Blas.tbmv(uplo, trans, diag, n, k, A, x)

    triangular banded matrix-vector `x = op(A) * x`

.. py:method:: numbapro.cudalib.cublas.Blas.tpmv(uplo, trans, diag, n, AP, x)

    triangular packed matrix-vector multiplication `x = op(A) * x`

.. py:method:: numbapro.cudalib.cublas.Blas.trsv(uplo, trans, diag, n, A, x)

    Solves the triangular linear system with a single right-hand-side.
    `op(A) * x = b`

.. py:method:: numbapro.cudalib.cublas.Blas.tpsv(uplo, trans, diag, n, AP, x)

    Solves the packed triangular linear system with a single right-hand-side.
    `op(A) * x = b`

.. py:method:: numbapro.cudalib.cublas.Blas.tbsv(uplo, trans, diag, n, k, A, x)

    Solves the triangular banded linear system with a single right-hand-side.
    `op(A) * x = b`

.. py:method:: numbapro.cudalib.cublas.Blas.symv(uplo, n, alpha, A, x, beta, y)

    symmetric matrix-vector multiplication `y = alpha * A * x + beta * y`

.. py:method:: numbapro.cudalib.cublas.Blas.hemv(uplo, n, alpha, A, x, beta, y)

    Hermitian matrix-vector multiplication `y = alpha * A * x + beta * y`

.. py:method:: numbapro.cudalib.cublas.Blas.sbmv(uplo, n, k, alpha, A, x, beta, y)

    symmetric banded matrix-vector multiplication  `y = alpha * A * x + beta * y`

.. py:method:: numbapro.cudalib.cublas.Blas.hbmv(uplo, n, k, alpha, A, x, beta, y)

    Hermitian banded matrix-vector multiplication  `y = alpha * A * x + beta * y`

.. py:method:: numbapro.cudalib.cublas.Blas.spmv(uplo, n, alpha, AP, x, beta, y)

    symmetric packed matrix-vector multiplication `y = alpha * A * x + beta * y`

.. py:method:: numbapro.cudalib.cublas.Blas.hpmv(uplo, n, alpha, AP, x, beta, y)

    Hermitian packed matrix-vector multiplication `y = alpha * A * x + beta * y`

.. py:method:: numbapro.cudalib.cublas.Blas.ger(m, n, alpha, x, y, A)

    the rank-1 update `A := alpha * x * y ** T + A`

.. py:method:: numbapro.cudalib.cublas.Blas.geru(m, n, alpha, x, y, A)

    the rank-1 update `A := alpha * x * y ** T + A`

.. py:method:: numbapro.cudalib.cublas.Blas.gerc(m, n, alpha, x, y, A)

    the rank-1 update `A := alpha * x * y ** H + A`

.. py:method:: numbapro.cudalib.cublas.Blas.syr(uplo, n, alpha, x, A)

    symmetric rank 1 operation `A := alpha * x * x ** T + A`

.. py:method:: numbapro.cudalib.cublas.Blas.her(uplo, n, alpha, x, A)

    hermitian rank 1 operation  `A := alpha * x * x ** H + A`

.. py:method:: numbapro.cudalib.cublas.Blas.spr(uplo, n, alpha, x, AP)

    the symmetric rank 1 operation `A := alpha * x * x ** T + A`

.. py:method:: numbapro.cudalib.cublas.Blas.hpr(uplo, n, alpha, x, AP)

    hermitian rank 1 operation `A := alpha * x * x ** H + A`

.. py:method:: numbapro.cudalib.cublas.Blas.syr2(uplo, n, alpha, x, y, A)

    symmetric rank-2 update `A = alpha * x * y ** T + y * x ** T + A`

.. py:method:: numbapro.cudalib.cublas.Blas.her2(uplo, n, alpha, x, y, A)

    Hermitian rank-2 update `A = alpha * x * y ** H + alpha * y * x ** H + A`

.. py:method:: numbapro.cudalib.cublas.Blas.spr2(uplo, n, alpha, x, y, A)

    packed symmetric rank-2 update `A = alpha * x * y ** T + y * x ** T + A`

.. py:method:: numbapro.cudalib.cublas.Blas.hpr2(uplo, n, alpha, x, y, A)

    packed Hermitian rank-2 update `A = alpha * x * y ** H + alpha * y * x ** H + A`

BLAS Level 3
-------------

All level 3 routines follow the same naming convention for arguments as in
level 2 routines.

.. py:method:: numbapro.cudalib.cublas.Blas.gemm(transa, transb, m, n, k, alpha, A, B, beta, C)

    matrix-matrix multiplication `C = alpha * op(A) * op(B) + beta * C`

.. py:method:: numbapro.cudalib.cublas.Blas.syrk(uplo, trans, n, k, alpha, A, beta, C)

    symmetric rank- k update `C = alpha * op(A) * op(A) ** T + beta * C`

.. py:method:: numbapro.cudalib.cublas.Blas.herk(uplo, trans, n, k, alpha, A, beta, C)

    Hermitian rank- k update `C = alpha * op(A) * op(A) ** H + beta * C`

.. py:method:: numbapro.cudalib.cublas.Blas.symm(side, uplo, m, n, alpha, A, B, beta, C)

    symmetric matrix-matrix multiplication::

        if  side == 'L':
            C = alpha * A * B + beta * C
        else:  # side == 'R'
            C = alpha * B * A + beta * C

.. py:method:: numbapro.cudalib.cublas.Blas.hemm(side, uplo, m, n, alpha, A, B, beta, C)

    Hermitian matrix-matrix multiplication::

            if  side == 'L':
                C = alpha * A * B + beta * C
            else:   #  side == 'R':
                C = alpha * B * A + beta * C

.. py:method:: numbapro.cudalib.cublas.Blas.trsm(side, uplo, trans, diag, m, n, alpha, A, B)

    Solves the triangular linear system with multiple right-hand-sides::

        if  side == 'L':
            op(A) * X = alpha * B
        else:       # side == 'R'
            X * op(A) = alpha * B


.. py:method:: numbapro.cudalib.cublas.Blas.trmm(side, uplo, trans, diag, m, n, alpha, A, B, C)

    triangular matrix-matrix multiplication::

        if  side == ':'
            C = alpha * op(A) * B
        else:   # side == 'R'
            C = alpha * B * op(A)

.. py:method:: numbapro.cudalib.cublas.Blas.dgmm(side, m, n, A, x, C)

    matrix-matrix multiplication::

        if  mode == 'R':
            C = A * x * diag(X)
        else:       # mode == 'L'
            C = diag(X) * x * A


.. py:method:: numbapro.cudalib.cublas.Blas.geam(transa, transb, m, n, alpha, A, beta, B, C)

    matrix-matrix addition/transposition `C = alpha * op(A) + beta * op(B)`

