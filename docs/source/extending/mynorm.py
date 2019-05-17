import numpy as np
from numba import njit, types
from numba.extending import overload, register_jitable
from numba.errors import TypingError

import scipy.linalg


@register_jitable
def _oneD_norm_2(a):
    val = np.abs(a)
    return np.sqrt(np.sum(val * val))


@overload(scipy.linalg.norm)
def jit_norm(a, ord=None):
    if not isinstance(a, types.Array):
        raise TypingError("Only accepts NumPy ndarray")
    if not isinstance(a.dtype, (types.Integer, types.Float)):
        raise TypingError("Only integer and floating point types accpeted")
    if a.ndim not in [1, 2]:
        raise TypingError('3D and beyond are not allowed')
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
    elif a.ndim == 2:
        def _two_D_norm_2(a, ord=None):
            return _oneD_norm_2(a.ravel())
        return _two_D_norm_2


@njit
def use(a, ord=None):
    return scipy.linalg.norm(a, ord)


if __name__ == "__main__":
    a = np.arange(10)
    print(use(a))
    print(scipy.linalg.norm(a))
    b = np.arange(9).reshape((3, 3))
    print(use(b))
    print(scipy.linalg.norm(b))
