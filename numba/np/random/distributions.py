"""
Algorithmic implementations for generating different types
of random distributions.
"""

import numpy as np
import platform

from numba.core.config import IS_32BITS
from numba.np.random._constants import (wi_double, ki_double,
                                        ziggurat_nor_r, fi_double,
                                        wi_float, ki_float,
                                        ziggurat_nor_inv_r_f,
                                        ziggurat_nor_r_f, fi_float,
                                        we_double, ke_double,
                                        ziggurat_exp_r, fe_double,
                                        we_float, ke_float,
                                        ziggurat_exp_r_f, fe_float,
                                        INT64_MAX, ziggurat_nor_inv_r,
                                        M_PI)
from numba.np.random.generator_core import (next_double, next_float,
                                            next_uint32, next_uint64)
from numba import float32, int64, njit
from numba.np.numpy_support import numpy_version
# All of the following implementations are direct translations from:
# https://github.com/numpy/numpy/blob/7cfef93c77599bd387ecc6a15d186c5a46024dac/numpy/random/src/distributions/distributions.c


if IS_32BITS or platform.machine() in ['ppc64le', 'aarch64']:
    fastmath_args = {'contract':True}
else:
    fastmath_args = {}


if numpy_version >= (1, 21):
    @njit
    def np_log1p(x):
        return np.log1p(x)

    @njit
    def np_log1pf(x):
        return np.log1p(float32(x))

    @njit
    def random_rayleigh(bitgen, mode):
        return mode * np.sqrt(2.0 * random_standard_exponential(bitgen))
else:
    @njit
    def np_log1p(x):
        return np.log(1.0 + x)

    @njit
    def np_log1pf(x):
        f32_one = np.float32(1.0)
        return np.log(f32_one + float32(x))

    @njit
    def random_rayleigh(bitgen, mode):
        return mode * np.sqrt(-2.0 * np.log(1.0 - next_double(bitgen)))

if numpy_version >= (1, 22):
    @njit
    def np_expm1(x):
        return np.expm1(x)
else:
    @njit
    def np_expm1(x):
        return np.exp(x) - 1.0


@njit
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
                xx = -ziggurat_nor_inv_r * np.log1p(-next_double(bitgen))
                yy = -np.log1p(-next_double(bitgen))
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


@njit
def random_standard_normal_f(bitgen):
    while 1:
        r = next_uint32(bitgen)
        idx = r & 0xff
        sign = (r >> 8) & 0x1
        rabs = (r >> 9) & 0x0007fffff
        x = float32(float32(rabs) * wi_float[idx])
        if (sign & 0x1):
            x = -x
        if (rabs < ki_float[idx]):
            return x
        if (idx == 0):
            while 1:
                xx = float32(-ziggurat_nor_inv_r_f *
                             np_log1pf(-next_float(bitgen)))
                yy = float32(-np_log1pf(-next_float(bitgen)))
                if (float32(yy + yy) > float32(xx * xx)):
                    if ((rabs >> 8) & 0x1):
                        return -float32(ziggurat_nor_r_f + xx)
                    else:
                        return float32(ziggurat_nor_r_f + xx)
        else:
            if (((fi_float[idx - 1] - fi_float[idx]) * next_float(bitgen) +
                 fi_float[idx]) < float32(np.exp(-float32(0.5) * x * x))):
                return x


@njit
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
                return ziggurat_exp_r - np_log1p(-next_double(bitgen))
            elif ((fe_double[idx - 1] - fe_double[idx]) * next_double(bitgen) +
                  fe_double[idx] < np.exp(-x)):
                return x


@njit
def random_standard_exponential_f(bitgen):
    while 1:
        ri = next_uint32(bitgen)
        ri >>= 1
        idx = ri & 0xFF
        ri >>= 8
        x = float32(float32(ri) * we_float[idx])
        if (ri < ke_float[idx]):
            return x
        else:
            if (idx == 0):
                return float32(ziggurat_exp_r_f -
                               float32(np_log1pf(-next_float(bitgen))))
            elif ((fe_float[idx - 1] - fe_float[idx]) * next_float(bitgen) +
                  fe_float[idx] < float32(np.exp(float32(-x)))):
                return x


@njit
def random_standard_exponential_inv(bitgen):
    return -np_log1p(-next_double(bitgen))


@njit
def random_standard_exponential_inv_f(bitgen):
    return -np.log(float32(1.0) - next_float(bitgen))


@njit(fastmath=fastmath_args)
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


@njit(fastmath=fastmath_args)
def random_standard_gamma_f(bitgen, shape):
    f32_one = float32(1.0)
    shape = float32(shape)
    if (shape == f32_one):
        return random_standard_exponential_f(bitgen)
    elif (shape == float32(0.0)):
        return float32(0.0)
    elif (shape < f32_one):
        while 1:
            U = next_float(bitgen)
            V = random_standard_exponential_f(bitgen)
            if (U <= f32_one - shape):
                X = float32(pow(U, float32(f32_one / shape)))
                if (X <= V):
                    return X
            else:
                Y = float32(-np.log(float32((f32_one - U) / shape)))
                X = float32(pow(f32_one - shape + float32(shape * Y),
                            float32(f32_one / shape)))
                if (X <= (V + Y)):
                    return X
    else:
        b = shape - f32_one / float32(3.0)
        c = float32(f32_one / float32(np.sqrt(float32(9.0) * b)))
        while 1:
            while 1:
                X = float32(random_standard_normal_f(bitgen))
                V = float32(f32_one + c * X)
                if (V > float32(0.0)):
                    break

            V = float32(V * V * V)
            U = next_float(bitgen)
            if (U < f32_one - float32(0.0331) * (X * X) * (X * X)):
                return float32(b * V)

            if (np.log(U) < float32(0.5) * X * X + b *
                    (f32_one - V + np.log(V))):
                return float32(b * V)


@njit(fastmath=fastmath_args)
def random_normal(bitgen, loc, scale):
    scaled_normal = scale * random_standard_normal(bitgen)
    return loc + scaled_normal


@njit(fastmath=fastmath_args)
def random_normal_f(bitgen, loc, scale):
    scaled_normal = float32(scale * random_standard_normal_f(bitgen))
    return float32(loc + scaled_normal)


@njit
def random_exponential(bitgen, scale):
    return scale * random_standard_exponential(bitgen)


@njit(fastmath=fastmath_args)
def random_uniform(bitgen, lower, range):
    scaled_uniform = range * next_double(bitgen)
    return lower + scaled_uniform


@njit
def random_gamma(bitgen, shape, scale):
    return scale * random_standard_gamma(bitgen, shape)


@njit
def random_gamma_f(bitgen, shape, scale):
    return float32(scale * random_standard_gamma_f(bitgen, shape))


@njit(fastmath=fastmath_args)
def random_beta(bitgen, a, b):
    if a <= 1.0 and b <= 1.0:
        while 1:
            U = next_double(bitgen)
            V = next_double(bitgen)
            X = pow(U, 1.0 / a)
            Y = pow(V, 1.0 / b)
            XpY = X + Y
            if XpY <= 1.0 and XpY > 0.0:
                if (X + Y > 0):
                    return X / XpY
                else:
                    logX = np.log(U) / a
                    logY = np.log(V) / b
                    logM = min(logX, logY)
                    logX -= logM
                    logY -= logM

                    return np.exp(logX - np.log(np.exp(logX) + np.exp(logY)))
    else:
        Ga = random_standard_gamma(bitgen, a)
        Gb = random_standard_gamma(bitgen, b)
        return Ga / (Ga + Gb)


@njit
def random_chisquare(bitgen, df):
    return 2.0 * random_standard_gamma(bitgen, df / 2.0)


@njit
def random_f(bitgen, dfnum, dfden):
    return ((random_chisquare(bitgen, dfnum) * dfden) /
            (random_chisquare(bitgen, dfden) * dfnum))


@njit
def random_standard_cauchy(bitgen):
    return random_standard_normal(bitgen) / random_standard_normal(bitgen)


@njit
def random_pareto(bitgen, a):
    return np_expm1(random_standard_exponential(bitgen) / a)


@njit
def random_weibull(bitgen, a):
    if (a == 0.0):
        return 0.0
    return pow(random_standard_exponential(bitgen), 1. / a)


@njit
def random_power(bitgen, a):
    return pow(-np_expm1(-random_standard_exponential(bitgen)), 1. / a)


@njit(fastmath=fastmath_args)
def random_laplace(bitgen, loc, scale):
    U = next_double(bitgen)
    while U <= 0:
        U = next_double(bitgen)
    if (U >= 0.5):
        U = loc - scale * np.log(2.0 - U - U)
    elif (U > 0.0):
        U = loc + scale * np.log(U + U)
    return U


@njit(fastmath=fastmath_args)
def random_gumbel(bitgen, loc, scale):
    U = 1.0 - next_double(bitgen)
    while U >= 1.0:
        U = 1.0 - next_double(bitgen)
    return loc - scale * np.log(-np.log(U))


@njit(fastmath=fastmath_args)
def random_logistic(bitgen, loc, scale):
    U = next_double(bitgen)
    while U <= 0.0:
        U = next_double(bitgen)
    return loc + scale * np.log(U / (1.0 - U))


@njit
def random_lognormal(bitgen, mean, sigma):
    return np.exp(random_normal(bitgen, mean, sigma))


@njit
def random_standard_t(bitgen, df):
    num = random_standard_normal(bitgen)
    denom = random_standard_gamma(bitgen, df / 2)
    return np.sqrt(df / 2) * num / np.sqrt(denom)


@njit(fastmath=fastmath_args)
def random_wald(bitgen, mean, scale):
    mu_2l = mean / (2 * scale)
    Y = random_standard_normal(bitgen)
    Y = mean * Y * Y
    X = mean + mu_2l * (Y - np.sqrt(4 * scale * Y + Y * Y))
    U = next_double(bitgen)
    if (U <= mean / (mean + X)):
        return X
    else:
        return mean * mean / X


@njit(fastmath=fastmath_args)
def random_vonmises(bitgen, mu, kappa):
    if (kappa < 1e-8):
        return M_PI * (2 * next_double(bitgen) - 1)
    else:
        if (kappa < 1e-5):
            s = (1. / kappa + kappa)
        else:
            if (kappa <= 1e6):
                r = 1 + np.sqrt(1 + 4 * kappa * kappa)
                rho = (r - np.sqrt(2 * r)) / (2 * kappa)
                s = (1 + rho * rho) / (2 * rho)
            else:
                result = mu + np.sqrt(1. / kappa) * \
                    random_standard_normal(bitgen)
                if (result < -M_PI):
                    result += 2 * M_PI
                if (result > M_PI):
                    result -= 2 * M_PI
                return result

        while 1:
            U = next_double(bitgen)
            Z = np.cos(M_PI * U)
            W = (1 + s * Z) / (s + Z)
            Y = kappa * (s - W)
            V = next_double(bitgen)
            if ((Y * (2 - Y) - V >= 0) or (np.log(Y / V) + 1 - Y >= 0)):
                break

        U = next_double(bitgen)

        result = np.arccos(W)
        if (U < 0.5):
            result = -result
        result += mu
        neg = (result < 0)
        mod = np.fabs(result)
        mod = (np.fmod(mod + M_PI, 2 * M_PI) - M_PI)
        if (neg):
            mod *= -1

        return mod


@njit(fastmath=fastmath_args)
def random_geometric_search(bitgen, p):
    X = 1
    sum = prod = p
    q = 1.0 - p
    U = next_double(bitgen)
    while (U > sum):
        prod *= q
        sum += prod
        X = X + 1
    return X


@njit
def random_geometric_inversion(bitgen, p):
    return np.ceil(-random_standard_exponential(bitgen) / np.log1p(-p))


@njit
def random_geometric(bitgen, p):
    if (p >= 0.333333333333333333333333):
        return random_geometric_search(bitgen, p)
    else:
        return random_geometric_inversion(bitgen, p)


@njit(fastmath=fastmath_args)
def random_zipf(bitgen, a):
    am1 = a - 1.0
    b = pow(2.0, am1)
    while 1:
        U = 1.0 - next_double(bitgen)
        V = next_double(bitgen)
        X = np.floor(pow(U, -1.0 / am1))
        if (X > INT64_MAX or X < 1.0):
            continue

        T = pow(1.0 + 1.0 / X, am1)
        if (V * X * (T - 1.0) / (b - 1.0) <= T / b):
            return X


@njit(fastmath=fastmath_args)
def random_triangular(bitgen, left, mode,
                      right):
    base = right - left
    leftbase = mode - left
    ratio = leftbase / base
    leftprod = leftbase * base
    rightprod = (right - mode) * base

    U = next_double(bitgen)
    if (U <= ratio):
        return left + np.sqrt(U * leftprod)
    else:
        return right - np.sqrt((1.0 - U) * rightprod)


@njit(fastmath=fastmath_args)
def random_loggam(x):
    a = [8.333333333333333e-02, -2.777777777777778e-03,
         7.936507936507937e-04, -5.952380952380952e-04,
         8.417508417508418e-04, -1.917526917526918e-03,
         6.410256410256410e-03, -2.955065359477124e-02,
         1.796443723688307e-01, -1.39243221690590e+00]

    if ((x == 1.0) or (x == 2.0)):
        return 0.0
    elif (x < 7.0):
        n = int(7 - x)
    else:
        n = 0

    x0 = x + n
    x2 = (1.0 / x0) * (1.0 / x0)
    # /* log(2 * M_PI) */
    lg2pi = 1.8378770664093453e+00
    gl0 = a[9]

    for k in range(0, 9):
        gl0 *= x2
        gl0 += a[8 - k]

    gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * np.log(x0) - x0
    if (x < 7.0):
        for k in range(1, n + 1):
            gl = gl - np.log(x0 - 1.0)
            x0 = x0 - 1.0

    return gl


@njit(fastmath=fastmath_args)
def random_poisson_mult(bitgen, lam):
    enlam = np.exp(-lam)
    X = 0
    prod = 1.0
    while (1):
        U = next_double(bitgen)
        prod *= U
        if (prod > enlam):
            X += 1
        else:
            return X


@njit(fastmath=fastmath_args)
def random_poisson_ptrs(bitgen, lam):

    slam = np.sqrt(lam)
    loglam = np.log(lam)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    invalpha = 1.1239 + 1.1328 / (b - 3.4)
    vr = 0.9277 - 3.6224 / (b - 2)

    while (1):
        U = next_double(bitgen) - 0.5
        V = next_double(bitgen)
        us = 0.5 - np.fabs(U)
        k = int((2 * a / us + b) * U + lam + 0.43)
        if ((us >= 0.07) and (V <= vr)):
            return k

        if ((k < 0) or ((us < 0.013) and (V > us))):
            continue

        # /* log(V) == log(0.0) ok here */
        # /* if U==0.0 so that us==0.0, log is ok since always returns */
        if ((np.log(V) + np.log(invalpha) - np.log(a / (us * us) + b)) <=
           (-lam + k * loglam - random_loggam(k + 1))):
            return k


@njit(fastmath=fastmath_args)
def random_poisson(bitgen, lam):
    if (lam >= 10):
        return random_poisson_ptrs(bitgen, lam)
    elif (lam == 0):
        return 0
    else:
        return random_poisson_mult(bitgen, lam)


@njit
def random_negative_binomial(bitgen, n, p):
    Y = random_gamma(bitgen, n, (1 - p) / p)
    return random_poisson(bitgen, Y)


@njit
def random_noncentral_chisquare(bitgen, df, nonc):
    if np.isnan(nonc):
        return np.nan

    if nonc == 0:
        return random_chisquare(bitgen, df)

    if 1 < df:
        Chi2 = random_chisquare(bitgen, df - 1)
        n = random_standard_normal(bitgen) + np.sqrt(nonc)
        return Chi2 + n * n
    else:
        i = random_poisson(bitgen, nonc / 2.0)
        return random_chisquare(bitgen, df + 2 * i)


@njit
def random_noncentral_f(bitgen, dfnum, dfden, nonc):
    t = random_noncentral_chisquare(bitgen, dfnum, nonc) * dfden
    return t / (random_chisquare(bitgen, dfden) * dfnum)


@njit
def random_logseries(bitgen, p):
    r = np_log1p(-p)

    while 1:
        V = next_double(bitgen)
        if (V >= p):
            return 1
        U = next_double(bitgen)
        q = -np.expm1(r * U)
        if (V <= q * q):
            result = int64(np.floor(1 + np.log(V) / np.log(q)))
            if result < 1 or V == 0.0:
                continue
            else:
                return result
        if (V >= q):
            return 1
        else:
            return 2
