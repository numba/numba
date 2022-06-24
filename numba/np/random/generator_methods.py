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
from numba.np.random.distributions import \
    (random_standard_exponential_inv_f, random_standard_exponential_inv,
     random_standard_exponential, random_standard_normal_f,
     random_standard_gamma, random_standard_normal, random_uniform,
     random_standard_exponential_f, random_standard_gamma_f, random_normal,
     random_exponential, random_gamma)


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


def check_size(size):
    if not any([isinstance(size, UniTuple) and
                isinstance(size.dtype, types.Integer),
                isinstance(size, Tuple) and size.count == 0,
                isinstance(size, types.Integer)]):
        raise TypingError("Size argument either of None," +
                          " an integer, an empty tuple or a tuple of integers")


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
        check_size(size)

        def impl(inst, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator)
            return out
        return impl


# Overload the Generator().standard_exponential() method
@overload_method(types.NumPyRandomGeneratorType, 'standard_exponential')
def NumPyRandomGeneratorType_standard_exponential(inst, size=None,
                                                  dtype=np.float64,
                                                  method=None):
    if isinstance(method, types.Omitted):
        method = method.value

    # TODO: This way of selecting methods works practically but is
    # extremely hackish. we should try doing the default
    # method==inv comparision over here if possible
    if method:
        dist_func, nb_dt = _get_proper_func(
            random_standard_exponential_inv_f,
            random_standard_exponential_inv,
            dtype
        )
    else:
        dist_func, nb_dt = _get_proper_func(random_standard_exponential_f,
                                            random_standard_exponential,
                                            dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if is_nonelike(size):
        def impl(inst, size=None, dtype=np.float64, method=None):
            return nb_dt(dist_func(inst.bit_generator))
        return impl
    else:
        check_size(size)

        def impl(inst, size=None, dtype=np.float64, method=None):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator)
            return out
        return impl


# Overload the Generator().standard_normal() method
@overload_method(types.NumPyRandomGeneratorType, 'standard_normal')
def NumPyRandomGeneratorType_standard_normal(inst, size=None, dtype=np.float64):
    dist_func, nb_dt = _get_proper_func(random_standard_normal_f,
                                        random_standard_normal,
                                        dtype)
    if isinstance(size, types.Omitted):
        size = size.value

    if is_nonelike(size):
        def impl(inst, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator))
        return impl
    else:
        check_size(size)

        def impl(inst, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator)
            return out
        return impl


# Overload the Generator().standard_gamma() method
@overload_method(types.NumPyRandomGeneratorType, 'standard_gamma')
def NumPyRandomGeneratorType_standard_gamma(inst, shape, size=None,
                                            dtype=np.float64):
    dist_func, nb_dt = _get_proper_func(random_standard_gamma_f,
                                        random_standard_gamma,
                                        dtype)
    if isinstance(size, types.Omitted):
        size = size.value

    if is_nonelike(size):
        def impl(inst, shape, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, shape))
        return impl
    else:
        check_size(size)

        def impl(inst, shape, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, shape)
            return out
        return impl


# Overload the Generator().normal() method
@overload_method(types.NumPyRandomGeneratorType, 'normal')
def NumPyRandomGeneratorType_normal(inst, loc=0.0, scale=1.0,
                                    size=None):
    if isinstance(size, types.Omitted):
        size = size.value

    if is_nonelike(size):
        def impl(inst, loc=0.0, scale=1.0, size=None):
            return random_normal(inst.bit_generator, loc, scale)
        return impl
    else:
        check_size(size)

        def impl(inst, loc=0.0, scale=1.0, size=None):
            out = np.empty(size, dtype=np.float64)
            for i in np.ndindex(size):
                out[i] = random_normal(inst.bit_generator, loc, scale)
            return out
        return impl


# Overload the Generator().uniform() method
@overload_method(types.NumPyRandomGeneratorType, 'uniform')
def NumPyRandomGeneratorType_uniform(inst, low=0.0, high=1.0,
                                     size=None):
    if isinstance(size, types.Omitted):
        size = size.value

    if is_nonelike(size):
        def impl(inst, low=0.0, high=1.0, size=None):
            return random_uniform(inst.bit_generator, low, high - low)
        return impl
    else:
        check_size(size)

        def impl(inst, low=0.0, high=1.0, size=None):
            out = np.empty(size, dtype=np.float64)
            for i in np.ndindex(size):
                out[i] = random_uniform(inst.bit_generator, low, high - low)
            return out
        return impl


# Overload the Generator().exponential() method
@overload_method(types.NumPyRandomGeneratorType, 'exponential')
def NumPyRandomGeneratorType_exponential(inst, scale=1.0, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if is_nonelike(size):
        def impl(inst, scale=1.0, size=None):
            return random_exponential(inst.bit_generator, scale)
        return impl
    else:
        check_size(size)

        def impl(inst, scale=1.0, size=None):
            out = np.empty(size, dtype=np.float64)
            for i in np.ndindex(size):
                out[i] = random_exponential(inst.bit_generator, scale)
            return out
        return impl


# Overload the Generator().gamma() method
@overload_method(types.NumPyRandomGeneratorType, 'gamma')
def NumPyRandomGeneratorType_gamma(inst, shape, scale=1.0, size=None):

    if isinstance(size, types.Omitted):
        size = size.value

    if is_nonelike(size):
        def impl(inst, shape, scale=1.0, size=None):
            return random_gamma(inst.bit_generator, shape, scale)
        return impl
    else:
        check_size(size)

        def impl(inst, shape, scale=1.0, size=None):
            out = np.empty(size, dtype=np.float64)
            for i in np.ndindex(size):
                out[i] = random_gamma(inst.bit_generator, shape, scale)
            return out
        return impl
