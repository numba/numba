
import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
                                        ziggurat_nor_r, fi_double,
                                        wi_float, ki_float,
                                        ziggurat_nor_inv_r_f,
                                        ziggurat_nor_r_f, fi_float,
                                        we_double, ke_double,
                                        ziggurat_exp_r, fe_double,
                                        we_float, ke_float,
                                        ziggurat_exp_r_f, fe_float,
                                        ziggurat_nor_inv_r)
from numba.np.random.generator_core import (next_double, next_float,
                                            next_uint32, next_uint64)
from numba import float32
# All following implementations are direct translations from:
# https://github.com/numpy/numpy/blob/7cfef93c77599bd387ecc6a15d186c5a46024dac/numpy/random/src/distributions/distributions.c


@register_jitable
def random_standard_normal(bitgen):
    while 1:
        r = next_uint64(bitgen)
        idx = r & 0xff
        r >>= 8
        sign = r & 0x1
        rabs = (r >> 1) & 0x000fffffffffffff
        x = rabs * wi_double[idx]
        if (sign & 0x1):
            x = -x
        if rabs < ki_double[idx]:
            return x
        if idx == 0:
            while 1:
                xx = -ziggurat_nor_inv_r * np.log(1.0 - next_double(bitgen))
                yy = -np.log(1.0 - next_double(bitgen))
                if (yy + yy > xx * xx):
                    if ((rabs >> 8) & 0x1):
                        return -(ziggurat_nor_r + xx)
                    else:
                        return ziggurat_nor_r + xx
        else:
            if (((fi_double[idx - 1] - fi_double[idx]) *
                    next_double(bitgen) + fi_double[idx]) <
                    np.exp(-0.5 * x * x)):
                return x


@register_jitable
def random_standard_normal_f(bitgen):
    while 1:
        r = next_uint32(bitgen)
        idx = r & 0xff
        sign = (r >> 8) & 0x1
        rabs = (r >> 9) & 0x0007fffff
        x = float32(rabs * wi_float[idx])
        if (sign & 0x1):
            x = -x
        if (rabs < ki_float[idx]):
            return x
        if (idx == 0):
            while 1:
                xx = float32(-ziggurat_nor_inv_r_f *
                             np.log(1.0 - next_float(bitgen)))
                yy = float32(-np.log(1.0 - next_float(bitgen)))
                if (yy + yy > xx * xx):
                    if ((rabs >> 8) & 0x1):
                        return -float32((ziggurat_nor_r_f + xx))
                    else:
                        return float32(ziggurat_nor_r_f + xx)
        else:
            if (((fi_float[idx - 1] - fi_float[idx]) * next_float(bitgen) +
                 fi_float[idx]) < np.exp(-0.5 * x * x)):
                return x


@register_jitable
def random_standard_exponential(bitgen):
    while 1:
        ri = next_uint64(bitgen)
        ri >>= 3
        idx = ri & 0xFF
        ri >>= 8
        x = ri * we_double[idx]
        if (ri < ke_double[idx]):
            return x
        else:
            if idx == 0:
                return ziggurat_exp_r - np.log(1.0 - next_double(bitgen))
            elif ((fe_double[idx - 1] - fe_double[idx]) * next_double(bitgen) +
                  fe_double[idx] < np.exp(-x)):
                return x


@register_jitable
def random_standard_exponential_f(bitgen):
    while 1:
        ri = next_uint32(bitgen)
        ri >>= 1
        idx = ri & 0xFF
        ri >>= 8
        x = float32(ri * we_float[idx])
        if (ri < ke_float[idx]):
            return x
        else:
            if (idx == 0):
                return float32(ziggurat_exp_r_f -
                               float32(np.log(1.0 - next_float(bitgen))))
            elif ((fe_float[idx - 1] - fe_float[idx]) * next_float(bitgen) +
                  fe_float[idx] < np.exp(-x)):
                return x


@register_jitable
def random_standard_exponential_inv(bitgen):
    return -np.log(1.0 - next_double(bitgen))


@register_jitable
def random_standard_exponential_inv_f(bitgen):
    return -float32(np.log(1.0 - next_float(bitgen)))


@register_jitable
def random_standard_gamma(bitgen, shape):
    if (shape == 1.0):
        return random_standard_exponential(bitgen)
    elif (shape == 0.0):
        return 0.0
    elif (shape < 1.0):
        while 1:
            U = next_double(bitgen)
            V = random_standard_exponential(bitgen)
            if (U <= 1.0 - shape):
                X = pow(U, 1. / shape)
                if (X <= V):
                    return X
            else:
                Y = -np.log((1 - U) / shape)
                X = pow(1.0 - shape + shape * Y, 1. / shape)
                if (X <= (V + Y)):
                    return X
    else:
        b = shape - 1. / 3.
        c = 1. / np.sqrt(9 * b)
        while 1:
            while 1:
                X = random_standard_normal(bitgen)
                V = 1.0 + c * X
                if (V > 0.0):
                    break

            V = V * V * V
            U = next_double(bitgen)
            if (U < 1.0 - 0.0331 * (X * X) * (X * X)):
                return (b * V)

            if (np.log(U) < 0.5 * X * X + b * (1. - V + np.log(V))):
                return (b * V)


@register_jitable
def random_standard_gamma_f(bitgen, shape):
    if (shape == 1.0):
        return random_standard_exponential_f(bitgen)
    elif (shape == 0.0):
        return 0.0
    elif (shape < 1.0):
        while 1:
            U = next_float(bitgen)
            V = random_standard_exponential_f(bitgen)
            if (U <= 1.0 - shape):
                X = float32(pow(U, float32(1.0 / shape)))
                if (X <= V):
                    return X
            else:
                Y = float32(-np.log((1.0 - U) / shape))
                X = float32(pow(1.0 - shape + float32(shape * Y),
                            float32(1.0 / shape)))
                if (X <= (V + Y)):
                    return X
    else:
        b = float32(shape - float32(1.0 / 3.0))
        c = float32(1.0 / float32(np.sqrt(9.0 * b)))
        while 1:
            while 1:
                X = float32(random_standard_normal_f(bitgen))
                V = float32(1.0 + c * X)
                if (V > 0.0):
                    break

            V = float32(V * V * V)
            U = next_float(bitgen)
            if (U < 1.0 - 0.0331 * (X * X) * (X * X)):
                return float32(b * V)

            if (np.log(U) < 0.5 * X * X + b * (1.0 - V + np.log(V))):
                return float32(b * V)


@register_jitable
def random_normal(bitgen, loc, scale):
    return loc + scale * random_standard_normal(bitgen)


@register_jitable
def random_normal_f(bitgen, loc, scale):
    return loc + scale * random_standard_normal_f(bitgen)


@register_jitable
def random_exponential(bitgen, scale):
    return scale * random_standard_exponential(bitgen)


@register_jitable
def random_exponential_f(bitgen, scale):
    return scale * random_standard_exponential_f(bitgen)


@register_jitable
def random_uniform(bitgen, lower, range):
    return lower + range * next_double(bitgen)


@register_jitable
def random_uniform_f(bitgen, lower, range):
    return lower + range * next_float(bitgen)


@register_jitable
def random_gamma(bitgen, shape, scale):
    return scale * random_standard_gamma(bitgen, shape)


@register_jitable
def random_gamma_f(bitgen, shape, scale):
    return scale * random_standard_gamma_f(bitgen, shape)
