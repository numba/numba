import math

from llvmlite import ir

from numba import np
from numba.core.cgutils import is_nonelike, is_empty_tuple
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core.typing import signature
from numba.core import types, cgutils
from numba.core.errors import NumbaTypeError
from numba.cpython._binomial import make_binomial_btpe
from numba.np.random._constants import LONG_MAX

from numba.cpython.randomimpl import (
    get_state_ptr, get_next_double, _double_preprocessor,
    _randrange_preprocessor, _randrange_impl, uniform_impl,
    _betavariate_impl, _gammavariate_impl, _lognormvariate_impl,
    _vonmisesvariate_impl, _seed_impl, _gauss_impl, int64_t,
    double, rnd_state_ptr_t, do_shuffle_impl
)


def get_np_state_ptr(context, builder):
    """
    Get a pointer to the thread-local Numpy random state.
    """
    return get_state_ptr(context, builder, 'np')


@overload(np.random.seed)
def seed_impl_np(seed):
    if isinstance(seed, types.Integer):
        return _seed_impl('np')


@overload(np.random.random)
@overload(np.random.random_sample)
@overload(np.random.sample)
@overload(np.random.ranf)
def random_impl0():
    @intrinsic
    def _impl(typingcontext):
        def codegen(context, builder, sig, args):
            state_ptr = get_state_ptr(context, builder, "np")
            return get_next_double(context, builder, state_ptr)
        return signature(types.float64), codegen
    return lambda: _impl()


@overload(np.random.random)
@overload(np.random.random_sample)
@overload(np.random.sample)
@overload(np.random.ranf)
def random_impl1(size=None):
    if is_nonelike(size):
        return lambda size=None: np.random.random()
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda size=None: np.array(np.random.random())
    if isinstance(size, types.Integer) or (isinstance(size, types.UniTuple)
                                           and isinstance(size.dtype,
                                                          types.Integer)):
        def _impl(size=None):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.random()
            return out
        return _impl


@overload(np.random.standard_normal)
@overload(np.random.normal)
def np_gauss_impl0():
    return lambda: np.random.normal(0.0, 1.0)


@overload(np.random.normal)
def np_gauss_impl1(loc):
    if isinstance(loc, (types.Float, types.Integer)):
        return lambda loc: np.random.normal(loc, 1.0)


@overload(np.random.normal)
def np_gauss_impl2(loc, scale):
    if isinstance(loc, (types.Float, types.Integer)) and isinstance(
            scale, (types.Float, types.Integer)):
        @intrinsic
        def _impl(typingcontext, loc, scale):
            loc_preprocessor = _double_preprocessor(loc)
            scale_preprocessor = _double_preprocessor(scale)
            return (
                signature(types.float64, loc, scale),
                _gauss_impl("np", loc_preprocessor,
                            scale_preprocessor, np.random.random)
            )
        return lambda loc, scale: _impl(loc, scale)


@overload(np.random.standard_normal)
def standard_normal_impl1(size):
    if is_nonelike(size):
        return lambda size: np.random.standard_normal()
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda size: np.array(np.random.standard_normal())
    if isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                           isinstance(size.dtype,
                                                      types.Integer)):
        def _impl(size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.standard_normal()
            return out
        return _impl


@overload(np.random.normal)
def np_gauss_impl3(loc, scale, size):
    if (isinstance(loc, (types.Float, types.Integer)) and isinstance(
            scale, (types.Float, types.Integer)) and
       is_nonelike(size)):
        return lambda loc, scale, size: np.random.normal(loc, scale)
    if (isinstance(loc, (types.Float, types.Integer)) and isinstance(
            scale, (types.Float, types.Integer)) and
       is_empty_tuple(size)):
        # Handle size = ()
        return lambda loc, scale, size: np.array(np.random.normal(loc, scale))
    if (isinstance(loc, (types.Float, types.Integer)) and isinstance(
            scale, (types.Float, types.Integer)) and
       (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple)
                                            and isinstance(size.dtype,
                                                           types.Integer)))):
        def _impl(loc, scale, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.normal(loc, scale)
            return out
        return _impl


@overload(np.random.randint)
def np_randint_impl_1(low):
    if isinstance(low, types.Integer):
        return lambda low: np.random.randint(0, low)


@overload(np.random.randint)
def np_randint_impl_2(low, high):
    if isinstance(low, types.Integer) and isinstance(high, types.Integer):
        signed = max(low.signed, high.signed)
        bitwidth = max(low.bitwidth, high.bitwidth)
        int_ty = types.Integer.from_bitwidth(bitwidth, signed)
        llvm_type = ir.IntType(bitwidth)

        start_preprocessor = _randrange_preprocessor(bitwidth, low)
        stop_preprocessor = _randrange_preprocessor(bitwidth, high)

        @intrinsic
        def _impl(typingcontext, low, high):
            def codegen(context, builder, sig, args):
                start, stop = args

                start = start_preprocessor(builder, start, llvm_type)
                stop = stop_preprocessor(builder, stop, llvm_type)
                step = ir.Constant(llvm_type, 1)
                return _randrange_impl(context, builder, start, stop, step,
                                       llvm_type, signed, 'np')
            return signature(int_ty, low, high), codegen
        return lambda low, high: _impl(low, high)


@overload(np.random.randint)
def np_randint_impl_3(low, high, size):
    if (isinstance(low, types.Integer) and isinstance(high, types.Integer) and
       is_nonelike(size)):
        return lambda low, high, size: np.random.randint(low, high)
    if (isinstance(low, types.Integer) and isinstance(high, types.Integer) and
       is_empty_tuple(size)):
        # Handle size = ()
        return lambda low, high, size: np.array(np.random.randint(low, high))
    if (isinstance(low, types.Integer) and isinstance(high, types.Integer) and
       (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple)
                                            and isinstance(size.dtype,
                                                           types.Integer)))):
        bitwidth = max(low.bitwidth, high.bitwidth)
        result_type = getattr(np, f'int{bitwidth}')

        def _impl(low, high, size):
            out = np.empty(size, dtype=result_type)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.randint(low, high)
            return out
        return _impl


@overload(np.random.uniform)
def np_uniform_impl0():
    return lambda: np.random.uniform(0.0, 1.0)


@overload(np.random.uniform)
def np_uniform_impl2(low, high):
    if isinstance(low, (types.Float, types.Integer)) and isinstance(
            high, (types.Float, types.Integer)):
        @intrinsic
        def _impl(typingcontext, low, high):
            low_preprocessor = _double_preprocessor(low)
            high_preprocessor = _double_preprocessor(high)
            return signature(types.float64, low, high), uniform_impl(
                'np', low_preprocessor, high_preprocessor)
        return lambda low, high: _impl(low, high)


@overload(np.random.uniform)
def np_uniform_impl3(low, high, size):
    if (isinstance(low, (types.Float, types.Integer)) and isinstance(
            high, (types.Float, types.Integer)) and
       is_nonelike(size)):
        return lambda low, high, size: np.random.uniform(low, high)
    if (isinstance(low, (types.Float, types.Integer)) and isinstance(
            high, (types.Float, types.Integer)) and
       is_empty_tuple(size)):
        # When calling np.random.uniform with size = (), the returned value
        # isn't a float like when size = None. Instead, it's an array of
        # shape ()
        return lambda low, high, size: np.array(np.random.uniform(low, high))
    if (isinstance(low, (types.Float, types.Integer)) and isinstance(
            high, (types.Float, types.Integer)) and
       (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple)
                                            and isinstance(size.dtype,
                                                           types.Integer)))):
        def _impl(low, high, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.uniform(low, high)
            return out
        return _impl


@overload(np.random.triangular)
def triangular_impl_np_3(left, mode, right):
    if (isinstance(left, (types.Float, types.Integer)) and isinstance(
            mode, (types.Float, types.Integer)) and
            isinstance(right, (types.Float, types.Integer))):
        def _impl(left, mode, right):
            if right == left:
                return left
            u = np.random.random()
            c = (mode - left) / (right - left)
            if u > c:
                u = 1.0 - u
                c = 1.0 - c
                left, right = right, left
            return left + (right - left) * math.sqrt(u * c)

        return _impl


@overload(np.random.triangular)
def triangular_impl_np_4(left, mode, right, size=None):
    if is_nonelike(size):
        return lambda left, mode, right, size=None: np.random.triangular(left,
                                                                         mode,
                                                                         right)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda left, mode, right, size=None: np.array(
            np.random.triangular(left, mode, right)
        )
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                            isinstance(size.dtype,
                                                       types.Integer))):
        def _impl(left, mode, right, size=None):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.triangular(left, mode, right)
            return out
        return _impl


@overload(np.random.standard_gamma)
@overload(np.random.gamma)
def ol_np_random_gamma1(shape):
    if isinstance(shape, (types.Float, types.Integer)):
        return lambda shape: np.random.gamma(shape, 1.0)


@overload(np.random.gamma)
def ol_np_random_gamma2(shape, scale):
    if isinstance(shape, (types.Float, types.Integer)) and isinstance(
            scale, (types.Float, types.Integer)):
        fn = register_jitable(_gammavariate_impl(np.random.random))

        def impl(shape, scale):
            return fn(shape, scale)

        return impl


@overload(np.random.gamma)
def gamma_impl(shape, scale, size):
    if is_nonelike(size):
        return lambda shape, scale, size: np.random.gamma(shape, scale)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda shape, scale, size: np.array(
            np.random.gamma(shape, scale)
        )
    if isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                           isinstance(size.dtype,
                                                      types.Integer)):
        def _impl(shape, scale, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.gamma(shape, scale)
            return out
        return _impl


@overload(np.random.standard_gamma)
def standard_gamma_impl(shape, size):
    if is_nonelike(size):
        return lambda shape, size: np.random.standard_gamma(shape)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda shape, size: np.array(np.random.standard_gamma(shape))
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple)
                                            and isinstance(size.dtype,
                                                           types.Integer))):
        def _impl(shape, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.standard_gamma(shape)
            return out
        return _impl


@overload(np.random.beta)
def ol_np_random_beta(a, b):
    if isinstance(a, (types.Float, types.Integer)) and isinstance(
            b, (types.Float, types.Integer)):
        fn = register_jitable(_betavariate_impl(np.random.gamma))

        def impl(a, b):
            return fn(a, b)

        return impl


@overload(np.random.beta)
def beta_impl(a, b, size):
    if is_nonelike(size):
        return lambda a, b, size: np.random.beta(a, b)
    if is_empty_tuple(size):
        # When calling np.random.beta with size = (), the returned value isn't a
        # float like when size = None. Instead, it's an array of shape ()
        return lambda a, b, size: np.array(np.random.beta(a, b))
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple)
                                            and isinstance(size.dtype,
                                                           types.Integer))):
        def _impl(a, b, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.beta(a, b)
            return out
        return _impl


@overload(np.random.exponential)
def exponential_impl_1(scale):
    if isinstance(scale, (types.Float, types.Integer)):
        def _impl(scale):
            return -math.log(1.0 - np.random.random()) * scale
        return _impl


@overload(np.random.exponential)
def exponential_impl_2(scale, size):
    if is_nonelike(size):
        return lambda scale, size: np.random.exponential(scale)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda scale, size: np.array(np.random.exponential(scale))
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                            isinstance(size.dtype,
                                                       types.Integer))):
        def _impl(scale, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.exponential(scale)
            return out
        return _impl


@overload(np.random.standard_exponential)
@overload(np.random.exponential)
def exponential_impl_0():
    def _impl():
        return -math.log(1.0 - np.random.random())
    return _impl


@overload(np.random.standard_exponential)
def standard_exponential_impl(size):
    if is_nonelike(size):
        return lambda size: np.random.standard_exponential()
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda size: np.array(np.random.standard_exponential())
    if (isinstance(size, types.Integer) or
       (isinstance(size, types.UniTuple) and isinstance(size.dtype,
                                                        types.Integer))):
        def _impl(size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.standard_exponential()
            return out
        return _impl


@overload(np.random.lognormal)
def np_lognormal_impl0():
    return lambda: np.random.lognormal(0.0, 1.0)


@overload(np.random.lognormal)
def np_log_normal_impl1(mean):
    if isinstance(mean, (types.Float, types.Integer)):
        return lambda mean: np.random.lognormal(mean, 1.0)


@overload(np.random.lognormal)
def np_log_normal_impl2(mean, sigma):
    if isinstance(mean, (types.Float, types.Integer)) and isinstance(
            sigma, (types.Float, types.Integer)):
        fn = register_jitable(_lognormvariate_impl(np.random.normal))
        return lambda mean, sigma: fn(mean, sigma)


@overload(np.random.lognormal)
def lognormal_impl(mean, sigma, size):
    if is_nonelike(size):
        return lambda mean, sigma, size: np.random.lognormal(mean, sigma)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda mean, sigma, size: np.array(
            np.random.lognormal(mean, sigma)
        )
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                            isinstance(size.dtype,
                                                       types.Integer))):
        def _impl(mean, sigma, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.lognormal(mean, sigma)
            return out
        return _impl


@overload(np.random.pareto)
def pareto_impl(a):
    if isinstance(a, types.Float):
        def _impl(a):
            # Same as paretovariate() - 1.
            u = 1.0 - np.random.random()
            return 1.0 / u ** (1.0 / a) - 1

        return _impl


@overload(np.random.pareto)
def pareto_impl_2(a, size):
    if is_nonelike(size):
        return lambda a, size: np.random.pareto(a)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda a, size: np.array(np.random.pareto(a))
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                            isinstance(size.dtype,
                                                       types.Integer))):
        def _impl(a, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.pareto(a)
            return out
        return _impl


@overload(np.random.weibull)
def weibull_impl(a):
    if isinstance(a, (types.Float, types.Integer)):
        def _impl(a):
            # Same as weibullvariate(1.0, a)
            u = 1.0 - np.random.random()
            return (-math.log(u)) ** (1.0 / a)

        return _impl


@overload(np.random.weibull)
def weibull_impl2(a, size):
    if is_nonelike(size):
        return lambda a, size: np.random.weibull(a)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda a, size: np.array(np.random.weibull(a))
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                            isinstance(size.dtype,
                                                       types.Integer))):
        def _impl(a, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.weibull(a)
            return out
        return _impl


@overload(np.random.vonmises)
def vonmisesvariate_impl_np(mu, kappa):
    if isinstance(mu, types.Float) and isinstance(kappa, types.Float):
        return _vonmisesvariate_impl(np.random.random)


@overload(np.random.vonmises)
def vonmises_impl(mu, kappa, size):
    if is_nonelike(size):
        return lambda mu, kappa, size: np.random.vonmises(mu, kappa)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda mu, kappa, size: np.array(np.random.vonmises(mu, kappa))
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple)
                                            and isinstance(size.dtype,
                                                           types.Integer))):
        def _impl(mu, kappa, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.vonmises(mu, kappa)
            return out
        return _impl


@register_jitable
def _binomial_btpe_next_double(bitgen):
    # Legacy np.random path: the BTPE core's ``bitgen`` is unused; draw from
    # np.random's global MT19937 state.
    return np.random.random()


@register_jitable
def _binomial_btpe_squeeze(A, xm, m, n, y, r, q):
    # np.random's case-52 Stirling squeeze, frozen at NumPy's legacy constants
    # for RandomState stream reproducibility, do
    # not "correct" to the NumPy 2.5 Generator values. Keep in sync with NumPy's
    # legacy_random_binomial_btpe.
    x1 = y + 1
    f1 = m + 1
    z = n + 1 - m
    w = n - y + 1
    x2 = x1 * x1
    f2 = f1 * f1
    z2 = z * z
    w2 = w * w
    return A > (
        xm * math.log(f1 / x1) + (n - m + 0.5) * math.log(z / w) +
        (y - m) * math.log(w * r / (x1 * q)) +
        (13680. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2)
        / f1 / 166320. +
        (13680. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2)
        / z / 166320. +
        (13680. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2)
        / x1 / 166320. +
        (13680. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2)
        / w / 166320.)


_binomial_btpe = make_binomial_btpe(_binomial_btpe_next_double,
                                    _binomial_btpe_squeeze)


@overload(np.random.binomial)
def binomial_impl(n, p):
    if isinstance(n, types.Integer) and isinstance(
            p, (types.Float, types.Integer)):
        def _impl(n, p):
            """
            Binomial distribution.  Numpy's BINV algorithm is used for small
            means; Numpy's BTPE algorithm is used for large means.
            """
            if n < 0:
                raise ValueError("binomial(): n <= 0")
            if not (0.0 <= p <= 1.0):
                raise ValueError("binomial(): p outside of [0, 1]")
            if p == 0.0:
                return 0
            if p == 1.0:
                return n

            flipped = p > 0.5
            if flipped:
                p = 1.0 - p
            q = 1.0 - p

            np_prod = n * p
            if np_prod > 30.0:
                X = _binomial_btpe(0, n, p)  # bitgen arg unused for np.random
                return n - X if flipped else X

            niters = 1
            qn = q ** n
            while qn <= 1e-308:
                # Underflow => split into several iterations
                niters <<= 2
                n >>= 2
                qn = q ** n
                assert n > 0

            bound = min(n, np_prod + 10.0 * math.sqrt(np_prod * q + 1))

            total = 0
            while niters > 0:
                X = 0
                U = np.random.random()
                px = qn
                while X <= bound:
                    if U <= px:
                        total += n - X if flipped else X
                        niters -= 1
                        break
                    U -= px
                    X += 1
                    px = ((n - X + 1) * p * px) / (X * q)

            return total

        return _impl


@overload(np.random.binomial)
def binomial_impl_3(n, p, size):
    if is_nonelike(size):
        return lambda n, p, size: np.random.binomial(n, p)
    if is_empty_tuple(size):
        # When calling np.random.binomial with size = (),
        # the returned value isn't a float like when size = None.
        # Instead, it's an array of shape ()
        return lambda n, p, size: np.array(np.random.binomial(n, p))
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                            isinstance(size.dtype,
                                                       types.Integer))):
        def _impl(n, p, size):
            out = np.empty(size, dtype=np.intp)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.binomial(n, p)
            return out
        return _impl


@overload(np.random.chisquare)
def chisquare_impl(df):
    if isinstance(df, (types.Float, types.Integer)):
        def _impl(df):
            return 2.0 * np.random.standard_gamma(df / 2.0)

        return _impl


@overload(np.random.chisquare)
def chisquare_impl2(df, size):
    if is_nonelike(size):
        return lambda df, size: np.random.chisquare(df)
    if is_empty_tuple(size):
        return lambda df, size: np.array(np.random.chisquare(df))
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                            isinstance(size.dtype,
                                                       types.Integer))):
        def _impl(df, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.chisquare(df)
            return out
        return _impl


@overload(np.random.f)
def f_impl(dfnum, dfden):
    if isinstance(dfnum, (types.Float, types.Integer)) and isinstance(
            dfden, (types.Float, types.Integer)):
        def _impl(dfnum, dfden):
            return ((np.random.chisquare(dfnum) * dfden) /
                    (np.random.chisquare(dfden) * dfnum))

        return _impl


@overload(np.random.f)
def f_impl_3(dfnum, dfden, size):
    if (isinstance(dfnum, (types.Float, types.Integer)) and isinstance(
            dfden, (types.Float, types.Integer)) and
       is_nonelike(size)):
        return lambda dfnum, dfden, size: np.random.f(dfnum, dfden)
    if (isinstance(dfnum, (types.Float, types.Integer)) and isinstance(
            dfden, (types.Float, types.Integer)) and
       is_empty_tuple(size)):
        # Handle size = ()
        return lambda dfnum, dfden, size: np.array(np.random.f(dfnum, dfden))
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple)
                                            and isinstance(size.dtype,
                                                           types.Integer))):
        def _impl(dfnum, dfden, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.f(dfnum, dfden)
            return out
        return _impl


@overload(np.random.geometric)
def geometric_impl(p):
    if isinstance(p, (types.Float, types.Integer)):
        def _impl(p):
            # Numpy's algorithm.
            if p <= 0.0 or p > 1.0:
                raise ValueError("geometric(): p outside of (0, 1]")
            q = 1.0 - p
            if p >= 0.333333333333333333333333:
                X = int(1)
                sum = prod = p
                U = np.random.random()
                while U > sum:
                    prod *= q
                    sum += prod
                    X += 1
                return X
            else:
                return math.ceil(math.log(1.0 - np.random.random()) /
                                 math.log(q))

        return _impl


@overload(np.random.geometric)
def geometric_impl_2(p, size):
    if is_nonelike(size):
        return lambda p, size: np.random.geometric(p)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda p, size: np.array(np.random.geometric(p))
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                            isinstance(size.dtype,
                                                       types.Integer))):
        def _impl(p, size):
            out = np.empty(size, dtype=np.int64)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.geometric(p)
            return out
        return _impl


@overload(np.random.gumbel)
def gumbel_impl(loc, scale):
    if isinstance(loc, (types.Float, types.Integer)) and isinstance(
            scale, (types.Float, types.Integer)):
        def _impl(loc, scale):
            U = 1.0 - np.random.random()
            return loc - scale * math.log(-math.log(U))

        return _impl


@overload(np.random.gumbel)
def gumbel_impl3(loc, scale, size):
    if is_nonelike(size):
        return lambda loc, scale, size: np.random.gumbel(loc, scale)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda loc, scale, size: np.array(np.random.gumbel(loc, scale))
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple)
                                            and isinstance(size.dtype,
                                                           types.Integer))):
        def _impl(loc, scale, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.gumbel(loc, scale)
            return out
        return _impl


@overload(np.random.hypergeometric)
def hypergeometric_impl(ngood, nbad, nsample):
    if (isinstance(ngood, (types.Float, types.Integer)) and isinstance(
            nbad, (types.Float, types.Integer))
       and isinstance(nsample, (types.Float, types.Integer))):
        def _impl(ngood, nbad, nsample):
            """Numpy's algorithm for hypergeometric()."""
            d1 = int(nbad) + int(ngood) - int(nsample)
            d2 = float(min(nbad, ngood))

            Y = d2
            K = int(nsample)
            while Y > 0.0 and K > 0:
                Y -= math.floor(np.random.random() + Y / (d1 + K))
                K -= 1
            Z = int(d2 - Y)
            if ngood > nbad:
                return int(nsample) - Z
            else:
                return Z

        return _impl


@overload(np.random.hypergeometric)
def hypergeometric_impl_4(ngood, nbad, nsample, size):
    if is_nonelike(size):
        return lambda ngood, nbad, nsample, size:\
            np.random.hypergeometric(ngood, nbad, nsample)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda ngood, nbad, nsample, size:\
            np.array(np.random.hypergeometric(ngood, nbad, nsample))
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple)
                                            and isinstance(size.dtype,
                                                           types.Integer))):
        def _impl(ngood, nbad, nsample, size):
            out = np.empty(size, dtype=np.intp)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.hypergeometric(ngood, nbad, nsample)
            return out
        return _impl


@overload(np.random.laplace)
def laplace_impl0():
    return lambda: np.random.laplace(0.0, 1.0)


@overload(np.random.laplace)
def laplace_impl1(loc):
    if isinstance(loc, (types.Float, types.Integer)):
        return lambda loc: np.random.laplace(loc, 1.0)


@overload(np.random.laplace)
def laplace_impl2(loc, scale):
    if isinstance(loc, (types.Float, types.Integer)) and isinstance(
            scale, (types.Float, types.Integer)):
        return laplace_impl


@overload(np.random.laplace)
def laplace_impl3(loc, scale, size):
    if is_nonelike(size):
        return lambda loc, scale, size: np.random.laplace(loc, scale)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda loc, scale, size: np.array(np.random.laplace(loc, scale))
    if isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                           isinstance(size.dtype,
                                                      types.Integer)):
        def _impl(loc, scale, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.laplace(loc, scale)
            return out
        return _impl


def laplace_impl(loc, scale):
    U = np.random.random()
    if U < 0.5:
        return loc + scale * math.log(U + U)
    else:
        return loc - scale * math.log(2.0 - U - U)


@overload(np.random.logistic)
def logistic_impl0():
    return lambda: np.random.logistic(0.0, 1.0)


@overload(np.random.logistic)
def logistic_impl1(loc):
    if isinstance(loc, (types.Float, types.Integer)):
        return lambda loc: np.random.logistic(loc, 1.0)


@overload(np.random.logistic)
def logistic_impl2(loc, scale):
    if isinstance(loc, (types.Float, types.Integer)) and isinstance(
            scale, (types.Float, types.Integer)):
        return logistic_impl


@overload(np.random.logistic)
def logistic_impl3(loc, scale, size):
    if is_nonelike(size):
        return lambda loc, scale, size: np.random.logistic(loc, scale)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda loc, scale, size: np.array(np.random.logistic(loc, scale))
    if (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple)
                                            and isinstance(size.dtype,
                                                           types.Integer))):
        def _impl(loc, scale, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.logistic(loc, scale)
            return out
        return _impl


def logistic_impl(loc, scale):
    U = np.random.random()
    return loc + scale * math.log(U / (1.0 - U))


def _logseries_impl(p):
    """Numpy's algorithm for logseries()."""
    if p <= 0.0 or p > 1.0:
        raise ValueError("logseries(): p outside of (0, 1]")
    r = math.log(1.0 - p)

    while 1:
        V = np.random.random()
        if V >= p:
            return 1
        U = np.random.random()
        q = 1.0 - math.exp(r * U)
        if V <= q * q:
            # XXX what if V == 0.0 ?
            return np.int64(1.0 + math.log(V) / math.log(q))
        elif V >= q:
            return 1
        else:
            return 2


@overload(np.random.logseries)
def logseries_impl(p):
    if isinstance(p, (types.Float, types.Integer)):
        return _logseries_impl


@overload(np.random.logseries)
def logseries_impl_2(p, size):
    if is_nonelike(size):
        return lambda p, size: np.random.logseries(p)
    if is_empty_tuple(size):
        return lambda p, size: np.array(np.random.logseries(p))
    if isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                           isinstance(size.dtype,
                                                      types.Integer)):
        def _impl(p, size):
            out = np.empty(size, dtype=np.int64)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.logseries(p)
            return out
        return _impl


@overload(np.random.negative_binomial)
def negative_binomial_impl(n, p):
    if isinstance(n, (types.Float, types.Integer)) and isinstance(
            p,(types.Float, types.Integer)):
        def _impl(n, p):
            if n <= 0:
                raise ValueError("negative_binomial(): n <= 0")
            if p < 0.0 or p > 1.0:
                raise ValueError("negative_binomial(): p outside of [0, 1]")
            Y = np.random.gamma(n, (1.0 - p) / p)
            return np.random.poisson(Y)

        return _impl


@overload(np.random.poisson)
def poisson_impl0():
    return lambda: np.random.poisson(1.0)


@overload(np.random.poisson)
def poisson_impl1(lam):
    if isinstance(lam, (types.Float, types.Integer)):
        @intrinsic
        def _impl(typingcontext, lam):
            lam_preprocessor = _double_preprocessor(lam)

            def codegen(context, builder, sig, args):
                state_ptr = get_np_state_ptr(context, builder)

                retptr = cgutils.alloca_once(builder, int64_t, name="ret")
                bbcont = builder.append_basic_block("bbcont")
                bbend = builder.append_basic_block("bbend")

                lam, = args
                lam = lam_preprocessor(builder, lam)
                big_lam = builder.fcmp_ordered('>=', lam,
                                               ir.Constant(double, 10.0))
                with builder.if_then(big_lam):
                    # For lambda >= 10.0, we switch to a more accurate
                    # algorithm (see _random.c).
                    fnty = ir.FunctionType(int64_t, (rnd_state_ptr_t, double))
                    fn = cgutils.get_or_insert_function(builder.function.module,
                                                        fnty,
                                                        "numba_poisson_ptrs")
                    ret = builder.call(fn, (state_ptr, lam))
                    builder.store(ret, retptr)
                    builder.branch(bbend)

                builder.branch(bbcont)
                builder.position_at_end(bbcont)

                _random = np.random.random
                _exp = math.exp

                def poisson_impl(lam):
                    """Numpy's algorithm for poisson() on small *lam*.

                    This method is invoked only if the parameter lambda of the
                    distribution is small ( < 10 ). The algorithm used is
                    described in "Knuth, D. 1969. 'Seminumerical Algorithms.
                    The Art of Computer Programming' vol 2.
                    """
                    if lam < 0.0:
                        raise ValueError("poisson(): lambda < 0")
                    if lam == 0.0:
                        return 0
                    enlam = _exp(-lam)
                    X = 0
                    prod = 1.0
                    while 1:
                        U = _random()
                        prod *= U
                        if prod <= enlam:
                            return X
                        X += 1

                ret = context.compile_internal(builder, poisson_impl, sig, args)
                builder.store(ret, retptr)
                builder.branch(bbend)
                builder.position_at_end(bbend)
                return builder.load(retptr)
            return signature(types.int64, lam), codegen
        return lambda lam: _impl(lam)


@overload(np.random.poisson)
def poisson_impl2(lam, size):
    if isinstance(lam, (types.Float, types.Integer)) and is_nonelike(size):
        return lambda lam, size: np.random.poisson(lam)
    if isinstance(lam, (types.Float, types.Integer)) and is_empty_tuple(size):
        # Handle size = ()
        return lambda lam, size: np.array(np.random.poisson(lam))
    if isinstance(lam, (types.Float, types.Integer)) and (
        isinstance(size, types.Integer) or
       (isinstance(size, types.UniTuple) and isinstance(size.dtype,
                                                        types.Integer))):
        def _impl(lam, size):
            out = np.empty(size, dtype=np.intp)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.poisson(lam)
            return out
        return _impl


@overload(np.random.power)
def power_impl(a):
    if isinstance(a, (types.Float, types.Integer)):
        def _impl(a):
            if a <= 0.0:
                raise ValueError("power(): a <= 0")
            return math.pow(1 - math.exp(-np.random.standard_exponential()),
                            1. / a)

        return _impl


@overload(np.random.power)
def power_impl_2(a, size):
    if is_nonelike(size):
        return lambda a, size: np.random.power(a)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda a, size: np.array(np.random.power(a))
    if isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                           isinstance(size.dtype,
                                                      types.Integer)):
        def _impl(a, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.power(a)
            return out
        return _impl


@overload(np.random.rayleigh)
def rayleigh_impl0():
    return lambda: np.random.rayleigh(1.0)


@overload(np.random.rayleigh)
def rayleigh_impl1(scale):
    if isinstance(scale, (types.Float, types.Integer)):
        def impl(scale):
            if scale <= 0.0:
                raise ValueError("rayleigh(): scale <= 0")
            return scale * math.sqrt(-2.0 * math.log(1.0 - np.random.random()))
        return impl


@overload(np.random.rayleigh)
def rayleigh_impl2(scale, size):
    if is_nonelike(size):
        return lambda scale, size: np.random.rayleigh(scale)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda scale, size: np.array(np.random.rayleigh(scale))
    if isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                           isinstance(size.dtype,
                                                      types.Integer)):
        def _impl(scale, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.rayleigh(scale)
            return out
        return _impl


@overload(np.random.standard_cauchy)
def cauchy_impl():
    def _impl():
        return np.random.standard_normal() / np.random.standard_normal()

    return _impl


@overload(np.random.standard_cauchy)
def standard_cauchy_impl(size):
    if is_nonelike(size):
        return lambda size: np.random.standard_cauchy()
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda size: np.array(np.random.standard_cauchy())
    if isinstance(size, types.Integer) or (isinstance(size, types.UniTuple)
                                           and isinstance(size.dtype,
                                                          types.Integer)):
        def _impl(size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.standard_cauchy()
            return out
        return _impl


@overload(np.random.standard_t)
def standard_t_impl(df):
    if isinstance(df, (types.Float, types.Integer)):
        def _impl(df):
            N = np.random.standard_normal()
            G = np.random.standard_gamma(df / 2.0)
            X = math.sqrt(df / 2.0) * N / math.sqrt(G)
            return X

        return _impl


@overload(np.random.standard_t)
def standard_t_impl2(df, size):
    if is_nonelike(size):
        return lambda df, size: np.random.standard_t(df)
    if is_empty_tuple(size):
        return lambda df, size: np.array(np.random.standard_t(df))
    if isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                           isinstance(size.dtype,
                                                      types.Integer)):
        def _impl(df, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.standard_t(df)
            return out
        return _impl


@overload(np.random.wald)
def wald_impl(mean, scale):
    if isinstance(mean, types.Float) and isinstance(scale, types.Float):
        def _impl(mean, scale):
            if mean <= 0.0:
                raise ValueError("wald(): mean <= 0")
            if scale <= 0.0:
                raise ValueError("wald(): scale <= 0")
            mu_2l = mean / (2.0 * scale)
            Y = np.random.standard_normal()
            Y = mean * Y * Y
            X = mean + mu_2l * (Y - math.sqrt(4 * scale * Y + Y * Y))
            U = np.random.random()
            if U <= mean / (mean + X):
                return X
            else:
                return mean * mean / X

        return _impl


@overload(np.random.wald)
def wald_impl2(mean, scale, size):
    if is_nonelike(size):
        return lambda mean, scale, size: np.random.wald(mean, scale)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda mean, scale, size: np.array(np.random.wald(mean, scale))
    if isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                           isinstance(size.dtype,
                                                      types.Integer)):
        def _impl(mean, scale, size):
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.wald(mean, scale)
            return out
        return _impl


@overload(np.random.zipf)
def zipf_impl(a):
    if isinstance(a, types.Float):
        def _impl(a):
            if a <= 1.0:
                raise ValueError("zipf(): a <= 1")
            am1 = a - 1.0
            b = 2.0 ** am1
            while 1:
                U = 1.0 - np.random.random()
                V = np.random.random()
                X = int(math.floor(U ** (-1.0 / am1)))

                if (X > LONG_MAX or X < 1.0):
                    continue

                T = (1.0 + 1.0 / X) ** am1
                if X >= 1 and V * X * (T - 1.0) / (b - 1.0) <= (T / b):
                    return X

        return _impl


@overload(np.random.zipf)
def zipf_impl_2(a, size):
    if is_nonelike(size):
        return lambda a, size: np.random.zipf(a)
    if is_empty_tuple(size):
        # Handle size = ()
        return lambda a, size: np.array(np.random.zipf(a))
    if isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and
                                           isinstance(size.dtype,
                                                      types.Integer)):
        def _impl(a, size):
            out = np.empty(size, dtype=np.intp)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.zipf(a)
            return out
        return _impl


@overload(np.random.shuffle)
def shuffle_impl_np(x):
    return do_shuffle_impl(x, np.random.randint)


@overload(np.random.permutation)
def permutation_impl(x):
    if isinstance(x, types.Integer):
        def permutation_impl(x):
            y = np.arange(x)
            np.random.shuffle(y)
            return y
    elif isinstance(x, types.Array):
        def permutation_impl(x):
            arr_copy = x.copy()
            np.random.shuffle(arr_copy)
            return arr_copy
    else:
        permutation_impl = None
    return permutation_impl


# ------------------------------------------------------------------------
# Irregular aliases: np.random.rand, np.random.randn

@overload(np.random.rand)
def rand(*size):
    if len(size) == 0:
        # Scalar output
        def rand_impl(*size):
            return np.random.random()

    else:
        # Array output
        def rand_impl(*size):
            return np.random.random(size)

    return rand_impl


@overload(np.random.randn)
def randn(*size):
    if len(size) == 0:
        # Scalar output
        def randn_impl(*size):
            return np.random.standard_normal()

    else:
        # Array output
        def randn_impl(*size):
            return np.random.standard_normal(size)

    return randn_impl


# ------------------------------------------------------------------------
# np.random.choice

@overload(np.random.choice)
def choice(a, size=None, replace=True):

    if isinstance(a, types.Array):
        # choice() over an array population
        assert a.ndim == 1
        dtype = a.dtype

        @register_jitable
        def get_source_size(a):
            return len(a)

        @register_jitable
        def copy_source(a):
            return a.copy()

        @register_jitable
        def getitem(a, a_i):
            return a[a_i]

    elif isinstance(a, types.Integer):
        # choice() over an implied arange() population
        dtype = np.intp

        @register_jitable
        def get_source_size(a):
            return a

        @register_jitable
        def copy_source(a):
            return np.arange(a)

        @register_jitable
        def getitem(a, a_i):
            return a_i

    else:
        raise NumbaTypeError("np.random.choice() first argument should be "
                             "int or array, got %s" % (a,))

    if size in (None, types.none):
        def choice_impl(a, size=None, replace=True):
            """
            choice() implementation returning a single sample
            (note *replace* is ignored)
            """
            n = get_source_size(a)
            i = np.random.randint(0, n)
            return getitem(a, i)

    else:
        def choice_impl(a, size=None, replace=True):
            """
            choice() implementation returning an array of samples
            """
            n = get_source_size(a)
            if replace:
                out = np.empty(size, dtype)
                fl = out.flat
                for i in range(len(fl)):
                    j = np.random.randint(0, n)
                    fl[i] = getitem(a, j)
                return out
            else:
                # Note we have to construct the array to compute out.size
                # (`size` can be an arbitrary int or tuple of ints)
                out = np.empty(size, dtype)
                if out.size > n:
                    raise ValueError("Cannot take a larger sample than "
                                     "population when 'replace=False'")
                # Get a permuted copy of the source array
                # we need this implementation in order to get the
                # np.random.choice inside numba to match the output
                # of np.random.choice outside numba when np.random.seed
                # is set to the same value
                permuted_a = np.random.permutation(a)
                fl = out.flat
                for i in range(len(fl)):
                    fl[i] = permuted_a[i]
                return out

    return choice_impl


# ------------------------------------------------------------------------
# np.random.multinomial

@overload(np.random.multinomial)
def multinomial(n, pvals, size=None):

    dtype = np.intp

    @register_jitable
    def multinomial_inner(n, pvals, out):
        # Numpy's algorithm for multinomial()
        fl = out.flat
        sz = out.size
        plen = len(pvals)

        for i in range(0, sz, plen):
            # Loop body: take a set of n experiments and fill up
            # fl[i:i + plen] with the distribution of results.

            # Current sum of outcome probabilities
            p_sum = 1.0
            # Current remaining number of experiments
            n_experiments = n
            # For each possible outcome `j`, compute the number of results
            # with this outcome.  This is done by considering the
            # conditional probability P(X=j | X>=j) and running a binomial
            # distribution over the remaining number of experiments.
            for j in range(0, plen - 1):
                p_j = pvals[j]
                n_j = fl[i + j] = np.random.binomial(n_experiments, p_j / p_sum)
                n_experiments -= n_j
                if n_experiments <= 0:
                    # Note the output was initialized to zero
                    break
                p_sum -= p_j
            if n_experiments > 0:
                # The remaining experiments end up in the last bucket
                fl[i + plen - 1] = n_experiments

    if not isinstance(n, types.Integer):
        raise NumbaTypeError("np.random.multinomial(): n should be an "
                             "integer, got %s" % (n,))

    if not isinstance(pvals, (types.Sequence, types.Array)):
        raise NumbaTypeError("np.random.multinomial(): pvals should be an "
                             "array or sequence, got %s" % (pvals,))

    if size in (None, types.none):
        def multinomial_impl(n, pvals, size=None):
            """
            multinomial(..., size=None)
            """
            out = np.zeros(len(pvals), dtype)
            multinomial_inner(n, pvals, out)
            return out

    elif isinstance(size, types.Integer):
        def multinomial_impl(n, pvals, size=None):
            """
            multinomial(..., size=int)
            """
            out = np.zeros((size, len(pvals)), dtype)
            multinomial_inner(n, pvals, out)
            return out

    elif isinstance(size, types.BaseTuple):
        def multinomial_impl(n, pvals, size=None):
            """
            multinomial(..., size=tuple)
            """
            out = np.zeros(size + (len(pvals),), dtype)
            multinomial_inner(n, pvals, out)
            return out

    else:
        raise NumbaTypeError("np.random.multinomial(): size should be int or "
                             "tuple or None, got %s" % (size,))

    return multinomial_impl

# ------------------------------------------------------------------------
# np.random.dirichlet


@overload(np.random.dirichlet)
def dirichlet(alpha):
    if isinstance(alpha, (types.Sequence, types.Array)):
        def dirichlet_impl(alpha):
            out = np.empty(len(alpha))
            dirichlet_arr(alpha, out)
            return out
        return dirichlet_impl


@overload(np.random.dirichlet)
def dirichlet_2(alpha, size=None):
    if not isinstance(alpha, (types.Sequence, types.Array)):
        raise NumbaTypeError(
            "np.random.dirichlet(): alpha should be an "
            "array or sequence, got %s" % (alpha,)
        )

    if size in (None, types.none) or is_empty_tuple(size):

        def dirichlet_impl(alpha, size=None):
            out = np.empty(len(alpha))
            dirichlet_arr(alpha, out)
            return out

    elif isinstance(size, types.Integer):

        def dirichlet_impl(alpha, size=None):
            """
            dirichlet(..., size=int)
            """
            out = np.empty((size, len(alpha)))
            dirichlet_arr(alpha, out)
            return out

    elif isinstance(size, types.UniTuple) and isinstance(size.dtype,
                                                         types.Integer):
        def dirichlet_impl(alpha, size=None):
            """
            dirichlet(..., size=tuple)
            """
            out = np.empty(size + (len(alpha),))
            dirichlet_arr(alpha, out)
            return out

    else:
        raise NumbaTypeError(
            "np.random.dirichlet(): size should be int or "
            "tuple of ints or None, got %s" % size
        )

    return dirichlet_impl


@register_jitable
def dirichlet_arr(alpha, out):

    # Gamma distribution method to generate a Dirichlet distribution

    for a_val in iter(alpha):
        if a_val <= 0:
            raise ValueError("dirichlet: alpha must be > 0.0")

    a_len = len(alpha)
    size = out.size
    flat = out.flat
    for i in range(0, size, a_len):
        # calculate gamma random numbers per alpha specifications
        norm = 0  # use this to normalize every the group total to 1
        for k, w in enumerate(alpha):
            flat[i + k] = np.random.gamma(w, 1)
            norm += flat[i + k].item()
        for k, w in enumerate(alpha):
            flat[i + k] /= norm


# ------------------------------------------------------------------------
# np.random.noncentral_chisquare


@overload(np.random.noncentral_chisquare)
def noncentral_chisquare(df, nonc):
    if isinstance(df, (types.Float, types.Integer)) and isinstance(
            nonc, (types.Float, types.Integer)):
        def noncentral_chisquare_impl(df, nonc):
            validate_noncentral_chisquare_input(df, nonc)
            return noncentral_chisquare_single(df, nonc)

        return noncentral_chisquare_impl


@overload(np.random.noncentral_chisquare)
def noncentral_chisquare_3(df, nonc, size=None):
    if size in (None, types.none):
        def noncentral_chisquare_impl(df, nonc, size=None):
            validate_noncentral_chisquare_input(df, nonc)
            return noncentral_chisquare_single(df, nonc)
        return noncentral_chisquare_impl
    if is_empty_tuple(size):
        # Handle size = ()
        def noncentral_chisquare_impl(df, nonc, size=None):
            validate_noncentral_chisquare_input(df, nonc)
            return np.array(noncentral_chisquare_single(df, nonc))
        return noncentral_chisquare_impl
    elif isinstance(size, types.Integer) or (
        isinstance(size, types.UniTuple) and isinstance(size.dtype,
                                                        types.Integer)):

        def noncentral_chisquare_impl(df, nonc, size=None):
            validate_noncentral_chisquare_input(df, nonc)
            out = np.empty(size)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = noncentral_chisquare_single(df, nonc)
            return out
        return noncentral_chisquare_impl
    else:
        raise NumbaTypeError(
            "np.random.noncentral_chisquare(): size should be int or "
            "tuple of ints or None, got %s" % size
        )


@register_jitable
def noncentral_chisquare_single(df, nonc):
    # identical to numpy implementation from distributions.c
    # https://github.com/numpy/numpy/blob/c65bc212ec1987caefba0ea7efe6a55803318de9/numpy/random/src/distributions/distributions.c#L797

    if np.isnan(nonc):
        return np.nan

    if 1 < df:
        chi2 = np.random.chisquare(df - 1)
        n = np.random.standard_normal() + np.sqrt(nonc)
        return chi2 + n * n

    else:
        i = np.random.poisson(nonc / 2.0)
        return np.random.chisquare(df + 2 * i)


@register_jitable
def validate_noncentral_chisquare_input(df, nonc):
    if df <= 0:
        raise ValueError("df <= 0")
    if nonc < 0:
        raise ValueError("nonc < 0")
