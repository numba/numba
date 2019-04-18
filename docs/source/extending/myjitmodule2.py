import numpy as np
from numba import njit, types
from numba.extending import overload
from numba.errors import TypingError

import mymodule


@overload(mymodule.set_to_x)
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
        # (The arr argument is an array *type* and the type of the values is stored as
        # the 'dtype' attribute. The x argument on the other hand is a scalar *type*
        # and thus x is already the type here so those things can be compared.)
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
                    "only integer and floating-point types are allowed")
        else:
            # type mismatch
            raise TypingError("the types of the inputs do not match")
    elif isinstance(arr, types.BaseTuple):
        # custom error for tuple as input
        raise TypingError("tuple isn't allowed as input, use NumPy ndarray")

    # fall through, None returned as no suitable implementation was found

@njit
def myalgorithm(a, x):
    # algorithm code
    mymodule.set_to_x(a, x)
    # more algorithm code
