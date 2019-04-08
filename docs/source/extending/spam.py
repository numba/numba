import numpy as np
from numba import njit, types
from numba.extending import overload
from numba.errors import TypingError


import ham


@overload(ham.set_to_x)
def set_to_x_jit(arr, x):

    if not isinstance(arr, types.Array):
        return
    if not isinstance(x, types.Integer):
        return

    def set_to_x_impl(arr, x):
        arr[:] = x

    return set_to_x_impl


@overload(ham.set_to_x)
def set_to_x_jit_v2(arr, x):

    # implementation for integers
    def set_to_x_impl_int(arr, x):
        arr[:] = x

    # implementation for floating-point
    def set_to_x_impl_float(arr, x):
        if np.any(np.isnan(arr)):
            raise ValueError("no element of arr must be nan")
        arr[:] = x

    # check that it is an array
    if isinstance(arr, types.Array):
        # validate that arr and x have the same type
        if arr.dtype == x:
            if isinstance(x, types.Integer):
                # dispatch for integers
                return set_to_x_impl_int
            elif isinstance(x, types.Float):
                # dispatch for float
                return set_to_x_impl_float
            else:
                # must be some other type
                raise TypingError(
                    "only integer and floating-point types allowed")
        else:
            # type mismatch
            raise TypingError("the types of the input do not match")
    elif isinstance(arr, types.BaseTuple):
        # custom error for tuple as input
        raise TypingError("tuple isn't allowed as input, use numpy arrays")

    # fall through, None returned as no suitable implementation was found


@njit
def breakfast(a, x):
    ham.set_to_x(a, x)
