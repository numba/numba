import numpy as np
from numba.core import types
from numba.core.extending import overload_method
from numba.core.imputils import Registry
from numba.np.numpy_support import as_dtype, from_dtype
from numba.np.random.generator_core import next_float, next_double
from numba.np.random.distributions import \
    (random_standard_exponential_inv_fill_f, random_standard_exponential_inv,
     random_standard_exponential, random_standard_normal_f,
     random_standard_gamma, random_standard_normal,
     random_standard_exponential_f, random_normal_f,
     random_standard_gamma_f, random_normal, random_exponential_f, random_power,
     random_exponential, random_gamma_f, random_gamma, random_beta,
     random_f,random_chisquare,random_standard_cauchy,random_pareto,
     random_weibull, random_laplace, random_gumbel, random_logistic,
     random_lognormal, random_rayleigh, random_standard_t, random_wald,
     random_vonmises, random_geometric, random_zipf, random_triangular,
     random_poisson, random_negative_binomial)
from numba.np.random.random_methods import \
    (random_bounded_uint64_fill, random_bounded_uint32_fill,
     random_bounded_uint16_fill, random_bounded_uint8_fill,
     random_bounded_bool_fill, _randint_arg_check, _choice)
from numba.core.extending import register_jitable

registry = Registry('generator_methods')


def get_proper_func(func_32, func_64, dtype):
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
    else:
        next_func = func_64

    return next_func, nb_dt


# Overload the Generator().integers()
@overload_method(types.NumPyRandomGeneratorType, 'integers')
def NumPyRandomGeneratorType_integers(inst, low, high=None, size=None,
                                      dtype=np.int64, endpoint=False):

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(dtype, types.Omitted):
        dtype = dtype.value

    if not isinstance(dtype, types.Type):
        dt = np.dtype(dtype)
        nb_dt = from_dtype(dt)
        _dtype = dtype
    else:
        nb_dt = dtype
        _dtype = as_dtype(nb_dt)

    if _dtype == np.int32:
        int_func = random_bounded_uint32_fill
        lower_bound = -0x80000000
        upper_bound = 0x7FFFFFFF
    elif _dtype == np.int64:
        int_func = random_bounded_uint64_fill
        lower_bound = -0x8000000000000000
        upper_bound = 0x7FFFFFFFFFFFFFFF
    elif _dtype == np.int16:
        int_func = random_bounded_uint16_fill
        lower_bound = -0x8000
        upper_bound = 0xFFFF
    elif _dtype == np.int8:
        int_func = random_bounded_uint8_fill
        lower_bound = -0x80
        upper_bound = 0xFF
    elif _dtype == np.uint32:
        int_func = random_bounded_uint32_fill
        lower_bound = -0x80000000
        upper_bound = 0x7FFFFFFF
    elif _dtype == np.uint64:
        int_func = random_bounded_uint64_fill
        lower_bound = -0x8000000000000000
        upper_bound = 0x7FFFFFFFFFFFFFFF
    elif _dtype == np.uint16:
        int_func = random_bounded_uint16_fill
        lower_bound = -0x8000
        upper_bound = 0xFFFF
    elif _dtype == np.uint8:
        int_func = random_bounded_uint8_fill
        lower_bound = -0x80
        upper_bound = 0xFF
    elif _dtype == np.bool_:
        int_func = random_bounded_bool_fill
        lower_bound = -1
        upper_bound = 2
    else:
        raise TypeError('Unsupported dtype %r for integers' % _dtype)

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, low, high=None, size=None,
                 dtype=np.int64, endpoint=False):
            low, rng = _randint_arg_check(low, high, endpoint,
                                          lower_bound, upper_bound)
            mask = None
            return int_func(inst.bit_generator, low, rng, mask, 1, dtype)[0]
        return impl
    else:
        def impl(inst, low, high=None, size=None,
                 dtype=np.int64, endpoint=False):
            low, rng = _randint_arg_check(low, high, endpoint,
                                          lower_bound, upper_bound)
            mask = None
            return int_func(inst.bit_generator, low, rng, mask, size, dtype)
        return impl


# Overload the Generator().choice()
@overload_method(types.NumPyRandomGeneratorType, 'choice')
def NumPyRandomGeneratorType_choice(inst, a, size=None, replace=True,
                                    p=None, axis=0, shuffle=True):

    if isinstance(a, types.Integer):
        @register_jitable
        def array_maker(a):
            return np.arange(a)
    else:
        @register_jitable
        def array_maker(a):
            return a
    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, a, size=None, replace=True,
                 p=None, axis=0, shuffle=True):
            return _choice(inst, array_maker(a), 1,
                           replace, p, axis, shuffle)[0]
        return impl
    else:
        def impl(inst, a, size=None, replace=True,
                 p=None, axis=0, shuffle=True):
            return _choice(inst, array_maker(a), size,
                           replace, p, axis, shuffle)
        return impl


# Overload the Generator().bytes()
@overload_method(types.NumPyRandomGeneratorType, 'bytes')
def NumPyRandomGeneratorType_bytes(inst, length):
    _size = ((length - 1) // 4 + 1)
    # Interpret the uint32s as little-endian to convert them to bytes
    # consistently.
    return NumPyRandomGeneratorType_integers(inst, 0, 4294967296,
                                             size=_size, dtype=np.uint32)\
        .astype('<u4').tobytes()[:length]


# Overload the Generator().random()
@overload_method(types.NumPyRandomGeneratorType, 'random')
def NumPyRandomGeneratorType_random(inst, size=None, dtype=np.float64):
    dist_func, nb_dt = get_proper_func(next_float, next_double, dtype)
    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator))
        return impl
    else:
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
                                                  use_zig_method=False):
    if use_zig_method:
        dist_func, nb_dt = get_proper_func(
            random_standard_exponential_inv_fill_f,
            random_standard_exponential_inv,
            dtype
        )
    else:
        dist_func, nb_dt = get_proper_func(random_standard_exponential_f,
                                           random_standard_exponential,
                                           dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, size=None, dtype=np.float64, use_zig_method=False):
            return nb_dt(dist_func(inst.bit_generator))
        return impl
    else:
        def impl(inst, size=None, dtype=np.float64, use_zig_method=False):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator)
            return out
        return impl


# Overload the Generator().standard_normal() method
@overload_method(types.NumPyRandomGeneratorType, 'standard_normal')
def NumPyRandomGeneratorType_standard_normal(inst, size=None, dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_standard_normal_f,
                                       random_standard_normal,
                                       dtype)
    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator))
        return impl
    else:
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
    dist_func, nb_dt = get_proper_func(random_standard_gamma_f,
                                       random_standard_gamma,
                                       dtype)
    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, shape, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, shape))
        return impl
    else:
        def impl(inst, shape, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, shape)
            return out
        return impl


# Overload the Generator().normal() method
@overload_method(types.NumPyRandomGeneratorType, 'normal')
def NumPyRandomGeneratorType_normal(inst, loc=0.0, scale=1.0,
                                    size=None, dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_normal_f, random_normal,
                                       dtype)
    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, loc=0.0, scale=1.0, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, loc, scale))
        return impl
    else:
        def impl(inst, loc=0.0, scale=1.0, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, loc, scale)
            return out
        return impl


# Overload the Generator().exponential() method
@overload_method(types.NumPyRandomGeneratorType, 'exponential')
def NumPyRandomGeneratorType_exponential(inst, scale=1.0, size=None,
                                         dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_exponential_f, random_exponential,
                                       dtype)
    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, scale=1.0, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, scale))
        return impl
    else:
        def impl(inst, scale=1.0, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, scale)
            return out
        return impl


# Overload the Generator().gamma() method
@overload_method(types.NumPyRandomGeneratorType, 'gamma')
def NumPyRandomGeneratorType_gamma(inst, shape, scale=1.0, size=None,
                                   dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_gamma_f, random_gamma, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, shape, scale=1.0, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, shape, scale))
        return impl
    else:
        def impl(inst, shape, scale=1.0, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, shape, scale)
            return out
        return impl


# Overload the Generator().beta() method
@overload_method(types.NumPyRandomGeneratorType, 'beta')
def NumPyRandomGeneratorType_beta(inst, a, b, size=None,
                                  dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_beta, random_beta, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, a, b, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, a, b))
        return impl
    else:
        def impl(inst, a, b, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, a, b)
            return out
        return impl


# Overload the Generator().chisquare() method
@overload_method(types.NumPyRandomGeneratorType, 'f')
def NumPyRandomGeneratorType_f(inst, dfnum, dfden, size=None,
                               dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_f, random_f, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, dfnum, dfden, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, dfnum, dfden))
        return impl
    else:
        def impl(inst, dfnum, dfden, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, dfnum, dfden)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'chisquare')
def NumPyRandomGeneratorType_chisquare(inst, df, size=None,
                                       dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_chisquare,
                                       random_chisquare, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, df, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, df))
        return impl
    else:
        def impl(inst, df, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, df)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'standard_cauchy')
def NumPyRandomGeneratorType_standard_cauchy(inst, size=None,
                                             dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_standard_cauchy,
                                       random_standard_cauchy, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator))
        return impl
    else:
        def impl(inst, size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'pareto')
def NumPyRandomGeneratorType_pareto(inst, a,size=None,
                                    dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_pareto, random_pareto, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, a,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, a))
        return impl
    else:
        def impl(inst, a,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, a)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'weibull')
def NumPyRandomGeneratorType_weibull(inst, a,size=None,
                                     dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_weibull, random_weibull, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, a,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, a))
        return impl
    else:
        def impl(inst, a,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, a)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'power')
def NumPyRandomGeneratorType_power(inst, a,size=None,
                                   dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_power, random_power, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, a,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, a))
        return impl
    else:
        def impl(inst, a,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, a)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'laplace')
def NumPyRandomGeneratorType_laplace(inst, loc=0.0, scale=1.0,size=None,
                                     dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_laplace, random_laplace, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, loc=0.0, scale=1.0,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, loc,scale))
        return impl
    else:
        def impl(inst, loc=0.0, scale=1.0,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, loc,scale)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'gumbel')
def NumPyRandomGeneratorType_gumbel(inst, loc=0.0, scale=1.0,size=None,
                                    dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_gumbel, random_gumbel, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, loc=0.0, scale=1.0,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, loc,scale))
        return impl
    else:
        def impl(inst, loc=0.0, scale=1.0,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, loc,scale)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'logistic')
def NumPyRandomGeneratorType_logistic(inst, loc=0.0, scale=1.0,size=None,
                                      dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_logistic, random_logistic, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, loc=0.0, scale=1.0,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, loc,scale))
        return impl
    else:
        def impl(inst, loc=0.0, scale=1.0,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, loc,scale)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'lognormal')
def NumPyRandomGeneratorType_lognormal(inst, mean=0.0, sigma=1.0,size=None,
                                       dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_lognormal,
                                       random_lognormal, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, mean=0.0, sigma=1.0,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, mean,sigma))
        return impl
    else:
        def impl(inst, mean=0.0, sigma=1.0,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, mean,sigma)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'rayleigh')
def NumPyRandomGeneratorType_rayleigh(inst, scale=1.0,size=None,
                                      dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_rayleigh, random_rayleigh, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, scale=1.0,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, scale))
        return impl
    else:
        def impl(inst, scale=1.0,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, scale)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'standard_t')
def NumPyRandomGeneratorType_standard_t(inst, df,size=None,
                                        dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_standard_t,
                                       random_standard_t, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, df,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, df))
        return impl
    else:
        def impl(inst, df,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, df)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'wald')
def NumPyRandomGeneratorType_wald(inst, mean, scale,size=None,
                                  dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_wald, random_wald, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, mean, scale,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, mean, scale))
        return impl
    else:
        def impl(inst, mean, scale,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, mean, scale)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'vonmises')
def NumPyRandomGeneratorType_vonmises(inst, mu, kappa,size=None,
                                      dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_vonmises, random_vonmises, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, mu, kappa,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, mu, kappa))
        return impl
    else:
        def impl(inst, mu, kappa,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, mu, kappa)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'geometric')
def NumPyRandomGeneratorType_geometric(inst, p,size=None,
                                       dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_geometric, random_geometric,
                                       dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, p,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, p))
        return impl
    else:
        def impl(inst, p,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, p)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'zipf')
def NumPyRandomGeneratorType_zipf(inst, a,size=None,
                                  dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_zipf, random_zipf, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, a,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, a))
        return impl
    else:
        def impl(inst, a,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, a)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'triangular')
def NumPyRandomGeneratorType_triangular(inst, left, mode, right,size=None,
                                        dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_triangular,
                                       random_triangular, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, left, mode, right,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, left, mode, right))
        return impl
    else:
        def impl(inst, left, mode, right,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, left, mode, right)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'poisson')
def NumPyRandomGeneratorType_poisson(inst, lam ,size=None,
                                     dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_poisson,
                                       random_poisson, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst, lam ,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, lam))
        return impl
    else:
        def impl(inst, lam ,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, lam)
            return out
        return impl


@overload_method(types.NumPyRandomGeneratorType, 'negative_binomial')
def NumPyRandomGeneratorType_negative_binomial(inst, n, p, size=None,
                                               dtype=np.float64):
    dist_func, nb_dt = get_proper_func(random_negative_binomial,
                                       random_negative_binomial, dtype)

    if isinstance(size, types.Omitted):
        size = size.value

    if isinstance(size, (types.NoneType,)) or size is None:
        def impl(inst,  n, p ,size=None, dtype=np.float64):
            return nb_dt(dist_func(inst.bit_generator, n, p))
        return impl
    else:
        def impl(inst, n, p ,size=None, dtype=np.float64):
            out = np.empty(size, dtype=dtype)
            for i in np.ndindex(size):
                out[i] = dist_func(inst.bit_generator, n, p)
            return out
        return impl
