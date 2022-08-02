import numpy as np
from numba import njit, types
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError

import scipy.linalg


@register_jitable
def _oneD_norm_2(a):
    # re-usable implementation of the 2-norm
    val = np.abs(a)
    return np.sqrt(np.sum(val * val))


@overload(scipy.linalg.norm)
def jit_norm(a, ord=None):
    if isinstance(ord, types.Optional):
        ord = ord.type
    # Reject non integer, floating-point or None types for ord
    if not isinstance(ord, (types.Integer, types.Float, types.NoneType)):
        raise TypingError("'ord' must be either integer or floating-point")
    # Reject non-ndarray types
    if not isinstance(a, types.Array):
        raise TypingError("Only accepts NumPy ndarray")
    # Reject ndarrays with non integer or floating-point dtype
    if not isinstance(a.dtype, (types.Integer, types.Float)):
        raise TypingError("Only integer and floating point types accepted")
    # Reject ndarrays with unsupported dimensionality
    if not (0 <= a.ndim <= 2):
        raise TypingError('3D and beyond are not allowed')
    # Implementation for scalars/0d-arrays
    elif a.ndim == 0:
        return a.item()
    # Implementation for vectors
    elif a.ndim == 1:
        def _oneD_norm_x(a, ord=None):
            if ord == 2 or ord is None:
                return _oneD_norm_2(a)
            elif ord == np.inf:
                return np.max(np.abs(a))
            elif ord == -np.inf:
                return np.min(np.abs(a))
            elif ord == 0:
                return np.sum(a != 0)
            elif ord == 1:
                return np.sum(np.abs(a))
            else:
                return np.sum(np.abs(a)**ord)**(1. / ord)
        return _oneD_norm_x
    # Implementation for matrices
    elif a.ndim == 2:
        def _two_D_norm_2(a, ord=None):
            return _oneD_norm_2(a.ravel())
        return _two_D_norm_2


if __name__ == "__main__":
    @njit
    def use(a, ord=None):
        # simple test function to check that the overload works
        return scipy.linalg.norm(a, ord)

    # spot check for vectors
    a = np.arange(10)
    print(use(a))
    print(scipy.linalg.norm(a))

    # spot check for matrices
    b = np.arange(9).reshape((3, 3))
    print(use(b))
    print(scipy.linalg.norm(b))
