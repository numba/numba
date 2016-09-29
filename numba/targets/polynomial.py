"""
Implementation of operations involving polynomials.
"""

from __future__ import print_function, absolute_import, division

import numpy as np

from numba import types, jit
from numba.extending import overload
from numba import numpy_support as np_support


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
