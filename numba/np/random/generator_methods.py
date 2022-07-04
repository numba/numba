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

    np_dt = dtype
    if isinstance(dtype, type):
        nb_dt = from_dtype(np.dtype(dtype))
    elif isinstance(dtype, types.NumberClass):
        nb_dt = dtype
        np_dt = as_dtype(nb_dt)

    if np_dt not in [np.float32, np.float64]:
        raise TypingError("Argument dtype is not one of the" +
                          " expected type(s): " +
                          " np.float32 or np.float64")

    if np_dt == np.float32:
        next_func = func_32
    else:
        next_func = func_64

    return next_func, nb_dt


def check_size(size):
    if not any([isinstance(size, UniTuple) and
                isinstance(size.dtype, types.Integer),
                isinstance(size, Tuple) and size.count == 0,
                isinstance(size, types.Integer)]):
        raise TypingError("Argument size is not one of the" +
                          " expected type(s): " +
                          " an integer, an empty tuple or a tuple of integers")


def check_types(obj, type_list, arg_name):
    """
    Check if given object is one of the provided types.
    If not raises an TypeError
    """
    if isinstance(obj, types.Omitted):
        obj = obj.value

    if not isinstance(type_list, (list, tuple)):
        type_list = [type_list]

    if not any([isinstance(obj, _type) for _type in type_list]):
        raise TypingError(f"Argument {arg_name} is not one of the" +
                          f" expected type(s): {type_list}")


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
                                                  method='zig'):
    check_types(method, [types.UnicodeType, str], 'method')
    dist_func_inv, nb_dt = _get_proper_func(
        random_standard_exponential_inv_f,
        random_standard_exponential_inv,
        dtype
    )

    dist_func, nb_dt = _get_proper_func(random_standard_exponential_f,
                                        random_standard_exponential,
                                        dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if is_nonelike(size):
        def impl(inst, size=None, dtype=np.float64, method='zig'):
            if method == 'zig':
                return nb_dt(dist_func(inst.bit_generator))
            elif method == 'inv':
                return nb_dt(dist_func_inv(inst.bit_generator))
            else:
                raise ValueError("Method must be either 'zig' or 'inv'")
        return impl
    else:
        check_size(size)

        def impl(inst, size=None, dtype=np.float64, method='zig'):
            out = np.empty(size, dtype=dtype)
            if method == 'zig':
                for i in np.ndindex(size):
                    out[i] = dist_func(inst.bit_generator)
            elif method == 'inv':
                for i in np.ndindex(size):
                    out[i] = dist_func_inv(inst.bit_generator)
            else:
                raise ValueError("Method must be either 'zig' or 'inv'")
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
    check_types(shape, [types.Float, types.Integer, int, float], 'shape')
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
    check_types(loc, [types.Float, types.Integer, int, float], 'loc')
    check_types(scale, [types.Float, types.Integer, int, float], 'scale')
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
    check_types(low, [types.Float, types.Integer, int, float], 'low')
    check_types(high, [types.Float, types.Integer, int, float], 'high')
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
    check_types(scale, [types.Float, types.Integer, int, float], 'scale')
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
    check_types(shape, [types.Float, types.Integer, int, float], 'shape')
    check_types(scale, [types.Float, types.Integer, int, float], 'scale')
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
