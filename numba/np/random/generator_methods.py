"""
Implementation of method overloads for Generator objects.
"""

import numpy as np
from numba.core import types
from numba.core.extending import overload_method
from numba.np.numpy_support import as_dtype, from_dtype
from numba.np.random.generator_core import next_float, next_double
from numba.np.numpy_support import is_nonelike
from numba.core.errors import TypingError
from numba.core.types.containers import Tuple, UniTuple


def _get_proper_func(func_32, func_64, dtype, dist_name="the given"):
    """
        Most of the standard NumPy distributions that accept dtype argument
        only support either np.float32 or np.float64 as dtypes.

        This is a helper function that helps Numba select the proper underlying
        implementation according to provided dtype.
    """
    if isinstance(dtype, types.Omitted):
        dtype = dtype.value

    if not isinstance(dtype, types.Type):
        dt = np.dtype(dtype)
        nb_dt = from_dtype(dt)
        np_dt = dtype
    else:
        nb_dt = dtype
        np_dt = as_dtype(nb_dt)

    np_dt = np.dtype(np_dt)

    if np_dt == np.float32:
        next_func = func_32
    elif np_dt == np.float64:
        next_func = func_64
    else:
        raise TypingError(
            f"Unsupported dtype {np_dt} for {dist_name} distribution"
        )

    return next_func, nb_dt


# Overload the Generator().random()
@overload_method(types.NumPyRandomGeneratorType, 'random')
def NumPyRandomGeneratorType_random(inst, size=None, dtype=np.float64):
    dist_func, nb_dt = _get_proper_func(next_float, next_double,
                                        dtype, "random")
    if isinstance(size, types.Omitted):
        size = size.value

    if is_nonelike(size):
        def impl(inst, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator))
        return impl
    else:
        assert any([isinstance(size, UniTuple) and
                   isinstance(size.dtype, types.Integer),
                   isinstance(size, Tuple) and size.count == 0,
                   isinstance(size, types.Integer)]
                   ), "Size argument either of None,"\
            + " an integer, an empty tuple or a tuple of integers"

        def impl(inst, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator)
            return out
        return impl
