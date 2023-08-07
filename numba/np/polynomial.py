"""
Implementation of operations involving polynomials.
"""


import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu

from numba import typeof
from numba.core import types, errors
from numba.core.extending import overload
from numba.np.numpy_support import type_can_asarray, as_dtype, from_dtype

@overload(np.roots)
def roots_impl(p):

    # cast int vectors to float cf. numpy, this is a bit dicey as
    # the roots could be complex which will fail anyway
    ty = getattr(p, 'dtype', p)
    if isinstance(ty, types.Integer):
        cast_t = np.float64
    else:
        cast_t = as_dtype(ty)

    def roots_impl(p):
        # impl based on numpy:
        # https://github.com/numpy/numpy/blob/master/numpy/lib/polynomial.py

        if len(p.shape) != 1:
            raise ValueError("Input must be a 1d array.")

        non_zero = np.nonzero(p)[0]

        if len(non_zero) == 0:
            return np.zeros(0, dtype=cast_t)

        tz = len(p) - non_zero[-1] - 1

        # pull out the coeffs selecting between possible zero pads
        p = p[int(non_zero[0]):int(non_zero[-1]) + 1]

        n = len(p)
        if n > 1:
            # construct companion matrix, ensure fortran order
            # to give to eigvals, write to upper diag and then
            # transpose.
            A = np.diag(np.ones((n - 2,), cast_t), 1).T
            A[0, :] = -p[1:] / p[0]  # normalize
            roots = np.linalg.eigvals(A)
        else:
            roots = np.zeros(0, dtype=cast_t)

        # add in additional zeros on the end if needed
        if tz > 0:
            return np.hstack((roots, np.zeros(tz, dtype=cast_t)))
        else:
            return roots

    return roots_impl


@overload(pu.trimseq)
def polyutils_trimseq(seq):
    if not type_can_asarray(seq):
        msg = 'The argument "seq" must be array-like'
        raise errors.TypingError(msg)

    if np.ndim(seq) > 1:
        msg = 'Coefficient array is not 1-d'
        raise errors.NumbaValueError(msg)

    def impl(seq):
        if len(seq) == 0:
            return seq
        else:
            for i in range(len(seq) - 1, -1, -1):
                if seq[i] != 0:
                    break
            return seq[:i+1]

    return impl


@overload(poly.polyadd)
def numpy_polyadd(c1, c2):
    if not type_can_asarray(c1):
        msg = 'The argument "c1" must be array-like'
        raise errors.TypingError(msg)

    if not type_can_asarray(c2):
        msg = 'The argument "c2" must be array-like'
        raise errors.TypingError(msg)

    if np.ndim(c1) > 1 or np.ndim(c2) > 1:
        msg = 'Coefficient array is not 1-d'
        raise errors.NumbaValueError(msg)

    if isinstance(c1, types.Number):
        s1 = str(as_dtype(c1))
    elif isinstance(c1, types.Tuple):
        t = [as_dtype(ty) for ty in c1.types]
        s1 = str(np.result_type(*t))
    else:
        s1 = str(c1.dtype)

    if isinstance(c2, types.Number):
        s2 = str(as_dtype(c2))
    elif isinstance(c2, types.Tuple):
        t = [as_dtype(ty) for ty in c2.types]
        s2 = str(np.result_type(*t))
    else:
        s2 = str(c2.dtype)

    result_dtype = from_dtype(np.result_type(s1, s2, np.float64))

    def impl(c1, c2):
        c1 = np.asarray(c1)
        c2 = np.asarray(c2)
        arr1 = np.atleast_1d(c1).astype(result_dtype)
        arr2 = np.atleast_1d(c2).astype(result_dtype)
        diff = len(arr2) - len(arr1)
        if diff > 0:
            zr = np.zeros(diff)
            arr1 = np.concatenate((arr1, zr))
        if diff < 0:
            zr = np.zeros(-diff)
            arr2 = np.concatenate((arr2, zr))
        val = arr1 + arr2
        return pu.trimseq(val)

    return impl


@overload(poly.polysub)
def numpy_polysub(c1, c2):
    if not type_can_asarray(c1):
        msg = 'The argument "c1" must be array-like'
        raise errors.TypingError(msg)

    if not type_can_asarray(c2):
        msg = 'The argument "c2" must be array-like'
        raise errors.TypingError(msg)

    if np.ndim(c1) > 1 or np.ndim(c2) > 1:
        msg = 'Coefficient array is not 1-d'
        raise errors.NumbaValueError(msg)

    if isinstance(c1, types.Number):
        s1 = str(as_dtype(c1))
    elif isinstance(c1, types.Tuple):
        t = [as_dtype(ty) for ty in c1.types]
        s1 = str(np.result_type(*t))
    else:
        s1 = str(c1.dtype)

    if isinstance(c2, types.Number):
        s2 = str(as_dtype(c2))
    elif isinstance(c2, types.Tuple):
        t = [as_dtype(ty) for ty in c2.types]
        s2 = str(np.result_type(*t))
    else:
        s2 = str(c2.dtype)

    result_dtype = from_dtype(np.result_type(s1, s2, np.float64))

    def impl(c1, c2):
        c1 = np.asarray(c1)
        c2 = np.asarray(c2)
        arr1 = np.atleast_1d(c1).astype(result_dtype)
        arr2 = np.atleast_1d(c2).astype(result_dtype)
        diff = len(arr2) - len(arr1)
        if diff > 0:
            zr = np.zeros(diff)
            arr1 = np.concatenate((arr1, zr))
        if diff < 0:
            zr = np.zeros(-diff)
            arr2 = np.concatenate((arr2, zr))
        val = arr1 - arr2
        return pu.trimseq(val)

    return impl


@overload(poly.polymul)
def numpy_polymul(c1, c2):
    if not type_can_asarray(c1):
        msg = 'The argument "c1" must be array-like'
        raise errors.TypingError(msg)

    if not type_can_asarray(c2):
        msg = 'The argument "c2" must be array-like'
        raise errors.TypingError(msg)

    if np.ndim(c1) > 1 or np.ndim(c2) > 1:
        msg = 'Coefficient array is not 1-d'
        raise errors.NumbaValueError(msg)

    if isinstance(c1, types.Number):
        s1 = str(as_dtype(c1))
    elif isinstance(c1, types.Tuple):
        t = [as_dtype(ty) for ty in c1.types]
        s1 = str(np.result_type(*t))
    else:
        s1 = str(c1.dtype)

    if isinstance(c2, types.Number):
        s2 = str(as_dtype(c2))
    elif isinstance(c2, types.Tuple):
        t = [as_dtype(ty) for ty in c2.types]
        s2 = str(np.result_type(*t))
    else:
        s2 = str(c2.dtype)

    result_dtype = from_dtype(np.result_type(s1, s2, np.float64))

    def impl(c1, c2):
        c1 = np.asarray(c1)
        c2 = np.asarray(c2)
        arr1 = np.atleast_1d(c1)
        arr2 = np.atleast_1d(c2)
        val = np.convolve(arr1, arr2).astype(result_dtype)
        return pu.trimseq(val)

    return impl