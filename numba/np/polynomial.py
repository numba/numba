"""
Implementation of operations involving polynomials.
"""


import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu

from numba import jit
from numba.core import types, errors
from numba.core.extending import overload, register_jitable
from numba.np import numpy_support as np_support

@overload(np.roots)
def roots_impl(p):

    # cast int vectors to float cf. numpy, this is a bit dicey as
    # the roots could be complex which will fail anyway
    ty = getattr(p, 'dtype', p)
    if isinstance(ty, types.Integer):
        cast_t = np.float64
    else:
        cast_t = np_support.as_dtype(ty)

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


@register_jitable
def _trimseq(seq):
    """Remove small Poly series coefficients.

    Parameters
    ----------
    seq : sequence
        Sequence of Poly series coefficients. This routine fails for
        empty sequences.

    Returns
    -------
    series : sequence
        Subsequence with trailing zeros removed. If the resulting sequence
        would be empty, return the first element. The returned sequence may
        or may not be a view.

    Notes
    -----
    Do not lose the type info if the sequence contains unknown objects.

    """
    if len(seq) == 0:
        return seq
    else:
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] != 0:
                break
        return seq[:i+1]


@overload(pu.trimseq)
def polyutils_trimseq(seq):
    if not isinstance(seq, types.Array):
        msg = 'The argument "seq" must be an array'
        raise errors.TypingError(msg)
    
    def impl(seq):
        return _trimseq(seq)
    
    return impl


@overload(poly.polyadd)
def numpy_polyadd(a1, a2):
    if not isinstance(a1, types.Array):
        msg = 'The argument "a1" must be an array'
        raise errors.TypingError(msg)
    
    if not isinstance(a2, types.Array):
        msg = 'The argument "a2" must be an array'
        raise errors.TypingError(msg)
    
    def impl(a1, a2):
        arr1 = np.atleast_1d(a1).astype(np.float64)
        arr2 = np.atleast_1d(a2).astype(np.float64)
        diff = len(a2) - len(a1)
        if diff > 0:
            zr = np.zeros(diff) #, a1.dtype)
            arr1 = np.concatenate((a1, zr))
        if diff < 0:
            zr = np.zeros(-diff) #, a2.dtype)
            arr2 = np.concatenate((a2, zr))
        val = arr1 + arr2
        return _trimseq(val)
    
    return impl

@overload(poly.polysub)
def numpy_polysub(a1, a2):
    if not isinstance(a1, types.Array):
        msg = 'The argument "a1" must be an array'
        raise errors.TypingError(msg)
    
    if not isinstance(a2, types.Array):
        msg = 'The argument "a2" must be an array'
        raise errors.TypingError(msg)
    
    def impl(a1, a2):
        arr1 = np.atleast_1d(a1).astype(np.float64)
        arr2 = np.atleast_1d(a2).astype(np.float64)
        diff = len(a2) - len(a1)
        if diff > 0:
            zr = np.zeros(diff) #, a1.dtype)
            arr1 = np.concatenate((a1, zr))
        if diff < 0:
            zr = np.zeros(-diff) #, a2.dtype)
            arr2 = np.concatenate((a2, zr))
        val = arr1 - arr2
        return _trimseq(val)
    
    return impl