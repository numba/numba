"""
Implement the cmath module functions.
"""


import cmath
import math

from numba.core import types, cgutils
from numba.cpython import mathimpl
from numba.core.extending import overload


@overload(cmath.isnan)
def isnan_complex_impl(z):
    if not isinstance(z, types.Complex):
        return

    def impl(z):
        return math.isnan(z.real) or math.isnan(z.imag)
    return impl


@overload(cmath.isinf)
def isinf_complex_impl(z):
    if not isinstance(z, types.Complex):
        return

    def impl(z):
        return math.isinf(z.real) or math.isinf(z.imag)
    return impl


@overload(cmath.isfinite)
def isfinite_complex_impl(z):
    if not isinstance(z, types.Complex):
        return

    def impl(z):
        return math.isfinite(z.real) and math.isfinite(z.imag)
    return impl


@overload(cmath.rect)
def impl_cmath_rect(r, phi):
    if all([isinstance(typ, types.Float) for typ in [r, phi]]):
        def impl(r, phi):
            if not math.isfinite(phi):
                if not r:
                    # cmath.rect(0, phi={inf, nan}) = 0
                    return abs(r)
                if math.isinf(r):
                    # cmath.rect(inf, phi={inf, nan}) = inf + j phi
                    return complex(r, phi)
            real = math.cos(phi)
            imag = math.sin(phi)
            if real == 0. and math.isinf(r):
                # 0 * inf would return NaN, we want to keep 0 but xor the sign
                real /= r
            else:
                real *= r
            if imag == 0. and math.isinf(r):
                # ditto
                imag /= r
            else:
                imag *= r
            return complex(real, imag)
        return impl


NAN = float('nan')
INF = float('inf')


@overload(cmath.exp)
def exp_impl(z):
    if not isinstance(z, types.Complex):
        return

    def _exp_impl(z):
        """cmath.exp(x + y j)"""
        x, y = z.real, z.imag
        x_is_finite = math.isfinite(x)
        y_is_finite = math.isfinite(y)
        if x_is_finite:
            if y_is_finite:
                c = math.cos(y)
                s = math.sin(y)
                r = math.exp(x)
                return complex(r * c, r * s)
            else:
                return complex(NAN, NAN)
        elif math.isnan(x):
            if y:
                return complex(x, x)  # nan + j nan
            else:
                return complex(x, y)  # nan + 0j
        elif x > 0.0:
            # x == +inf
            if y_is_finite:
                real = math.cos(y)
                imag = math.sin(y)
                # Avoid NaNs if math.cos(y) or math.sin(y) == 0
                # (e.g. cmath.exp(inf + 0j) == inf + 0j)
                if real != 0:
                    real *= x
                if imag != 0:
                    imag *= x
                return complex(real, imag)
            else:
                return complex(x, NAN)
        else:
            # x == -inf
            if y_is_finite:
                r = math.exp(x)
                c = math.cos(y)
                s = math.sin(y)
                return complex(r * c, r * s)
            else:
                r = 0
                return complex(r, r)
    return _exp_impl


@overload(cmath.log)
def log_base_impl(x, base=None):
    if not isinstance(x, types.Complex):
        return

    if cgutils.is_nonelike(base):
        def impl(x, base=None):
            """cmath.log(z + y j)"""
            z, y = x.real, x.imag
            a = math.log(math.hypot(z, y))
            b = math.atan2(y, z)
            return complex(a, b)
        return impl
    else:
        def log_base(x, base=None):
            """cmath.log(x, base)"""
            return cmath.log(x) / cmath.log(base)
        return log_base


@overload(cmath.log10)
def impl_cmath_log10(z):
    if not isinstance(z, types.Complex):
        return

    LN_10 = 2.302585092994045684

    def log10_impl(z):
        """cmath.log10(z)"""
        z = cmath.log(z)
        # This formula gives better results on +/-inf than cmath.log(z, 10)
        # See http://bugs.python.org/issue22544
        return complex(z.real / LN_10, z.imag / LN_10)

    return log10_impl


@overload(cmath.phase)
def phase_impl(x):
    """cmath.phase(x + y j)"""

    if not isinstance(x, types.Complex):
        return

    def impl(x):
        return math.atan2(x.imag, x.real)
    return impl


@overload(cmath.polar)
def polar_impl(x):
    if not isinstance(x, types.Complex):
        return

    def impl(x):
        r, i = x.real, x.imag
        return math.hypot(r, i), math.atan2(i, r)
    return impl


@overload(cmath.sqrt)
def sqrt_impl(z):
    # We risk spurious overflow for components >= FLT_MAX / (1 + sqrt(2)).

    SQRT2 = 1.414213562373095048801688724209698079E0
    ONE_PLUS_SQRT2 = (1. + SQRT2)
    theargflt = z.underlying_float if isinstance(z, types.Complex) else z

    # Get a type specific maximum value so scaling for overflow is based on that
    MAX = mathimpl.DBL_MAX if theargflt.bitwidth == 64 else mathimpl.FLT_MAX
    # THRES will be double precision, should not impact typing as it's just
    # used for comparison, there *may* be a few values near THRES which
    # deviate from e.g. NumPy due to rounding that occurs in the computation
    # of this value in the case of a 32bit argument.
    THRES = MAX / ONE_PLUS_SQRT2

    def _sqrt_impl(z):
        """cmath.sqrt(z)"""
        # This is NumPy's algorithm, see npy_csqrt() in npy_math_complex.c.src
        a = z.real
        b = z.imag
        if a == 0.0 and b == 0.0:
            return complex(abs(b), b)
        if math.isinf(b):
            return complex(abs(b), b)
        if math.isnan(a):
            return complex(a, a)
        if math.isinf(a):
            if a < 0.0:
                return complex(abs(b - b), math.copysign(a, b))
            else:
                return complex(a, math.copysign(b - b, b))

        # The remaining special case (b is NaN) is handled just fine by
        # the normal code path below.

        # Scale to avoid overflow
        if abs(a) >= THRES or abs(b) >= THRES:
            a *= 0.25
            b *= 0.25
            scale = True
        else:
            scale = False
        # Algorithm 312, CACM vol 10, Oct 1967
        if a >= 0:
            t = math.sqrt((a + math.hypot(a, b)) * 0.5)
            real = t
            imag = b / (2 * t)
        else:
            t = math.sqrt((-a + math.hypot(a, b)) * 0.5)
            real = abs(b) / (2 * t)
            imag = math.copysign(t, b)
        # Rescale
        if scale:
            return complex(real * 2, imag)
        else:
            return complex(real, imag)

    return _sqrt_impl


@overload(cmath.cos)
def cos_impl(z):
    if not isinstance(z, types.Complex):
        return

    def _cos_impl(z):
        """cmath.cos(z) = cmath.cosh(z j)"""
        return cmath.cosh(complex(-z.imag, z.real))
    return _cos_impl


@overload(cmath.cosh)
def impl_cmath_cosh(z):
    if not isinstance(z, types.Complex):
        return

    def cosh_impl(z):
        """cmath.cosh(z)"""
        x = z.real
        y = z.imag
        if math.isinf(x):
            if math.isnan(y):
                # x = +inf, y = NaN => cmath.cosh(x + y j) = inf + Nan * j
                real = abs(x)
                imag = y
            elif y == 0.0:
                # x = +inf, y = 0 => cmath.cosh(x + y j) = inf + 0j
                real = abs(x)
                imag = y
            else:
                real = math.copysign(x, math.cos(y))
                imag = math.copysign(x, math.sin(y))
            if x < 0.0:
                # x = -inf => negate imaginary part of result
                imag = -imag
            return complex(real, imag)
        return complex(math.cos(y) * math.cosh(x),
                       math.sin(y) * math.sinh(x))
    return cosh_impl


@overload(cmath.sin)
def sin_impl(z):
    if not isinstance(z, types.Complex):
        return

    def _sin_impl(z):
        """cmath.sin(z) = -j * cmath.sinh(z j)"""
        r = cmath.sinh(complex(-z.imag, z.real))
        return complex(r.imag, -r.real)
    return _sin_impl


@overload(cmath.sinh)
def impl_cmath_sinh(z):
    if not isinstance(z, types.Complex):
        return

    def sinh_impl(z):
        """cmath.sinh(z)"""
        x = z.real
        y = z.imag
        if math.isinf(x):
            if math.isnan(y):
                # x = +/-inf, y = NaN => cmath.sinh(x + y j) = x + NaN * j
                real = x
                imag = y
            else:
                real = math.cos(y)
                imag = math.sin(y)
                if real != 0.:
                    real *= x
                if imag != 0.:
                    imag *= abs(x)
            return complex(real, imag)
        return complex(math.cos(y) * math.sinh(x),
                       math.sin(y) * math.cosh(x))
    return sinh_impl


@overload(cmath.tan)
def tan_impl(z):
    if not isinstance(z, types.Complex):
        return

    def _tan_impl(z):
        """cmath.tan(z) = -j * cmath.tanh(z j)"""
        r = cmath.tanh(complex(-z.imag, z.real))
        return complex(r.imag, -r.real)
    return _tan_impl


@overload(cmath.tanh)
def impl_cmath_tanh(z):
    if not isinstance(z, types.Complex):
        return

    def tanh_impl(z):
        """cmath.tanh(z)"""
        x = z.real
        y = z.imag
        if math.isinf(x):
            real = math.copysign(1., x)
            if math.isinf(y):
                imag = 0.
            else:
                imag = math.copysign(0., math.sin(2. * y))
            return complex(real, imag)
        # This is CPython's algorithm (see c_tanh() in cmathmodule.c).
        # XXX how to force float constants into single precision?
        tx = math.tanh(x)
        ty = math.tan(y)
        cx = 1. / math.cosh(x)
        txty = tx * ty
        denom = 1. + txty * txty
        return complex(
            tx * (1. + ty * ty) / denom,
            ((ty / denom) * cx) * cx)

    return tanh_impl


@overload(cmath.acos)
def acos_impl(z):
    if not isinstance(z, types.Complex):
        return

    LN_4 = math.log(4)
    THRES = mathimpl.FLT_MAX / 4

    def _acos_impl(z):
        """cmath.acos(z)"""
        # CPython's algorithm (see c_acos() in cmathmodule.c)
        if abs(z.real) > THRES or abs(z.imag) > THRES:
            # Avoid unnecessary overflow for large arguments
            # (also handles infinities gracefully)
            real = math.atan2(abs(z.imag), z.real)
            imag = math.copysign(
                math.log(math.hypot(z.real * 0.5, z.imag * 0.5)) + LN_4,
                -z.imag)
            return complex(real, imag)
        else:
            s1 = cmath.sqrt(complex(1. - z.real, -z.imag))
            s2 = cmath.sqrt(complex(1. + z.real, z.imag))
            real = 2. * math.atan2(s1.real, s2.real)
            imag = math.asinh(s2.real * s1.imag - s2.imag * s1.real)
            return complex(real, imag)
    return _acos_impl


@overload(cmath.acosh)
def impl_cmath_acosh(z):
    if not isinstance(z, types.Complex):
        return

    LN_4 = math.log(4)
    THRES = mathimpl.FLT_MAX / 4

    def acosh_impl(z):
        """cmath.acosh(z)"""
        # CPython's algorithm (see c_acosh() in cmathmodule.c)
        if abs(z.real) > THRES or abs(z.imag) > THRES:
            # Avoid unnecessary overflow for large arguments
            # (also handles infinities gracefully)
            real = math.log(math.hypot(z.real * 0.5, z.imag * 0.5)) + LN_4
            imag = math.atan2(z.imag, z.real)
            return complex(real, imag)
        else:
            s1 = cmath.sqrt(complex(z.real - 1., z.imag))
            s2 = cmath.sqrt(complex(z.real + 1., z.imag))
            real = math.asinh(s1.real * s2.real + s1.imag * s2.imag)
            imag = 2. * math.atan2(s1.imag, s2.real)
            return complex(real, imag)
        # Condensed formula (NumPy)
        #return cmath.log(z + cmath.sqrt(z + 1.) * cmath.sqrt(z - 1.))

    return acosh_impl


@overload(cmath.asinh)
def asinh_impl(z):
    if not isinstance(z, types.Complex):
        return

    THRES = mathimpl.FLT_MAX / 4

    def _asinh_impl(z):
        """cmath.asinh(z)"""
        LN_4 = math.log(4)
        # CPython's algorithm (see c_asinh() in cmathmodule.c)
        if abs(z.real) > THRES or abs(z.imag) > THRES:
            real = math.copysign(
                math.log(math.hypot(z.real * 0.5, z.imag * 0.5)) + LN_4,
                z.real)
            imag = math.atan2(z.imag, abs(z.real))
            return complex(real, imag)
        else:
            s1 = cmath.sqrt(complex(1. + z.imag, -z.real))
            s2 = cmath.sqrt(complex(1. - z.imag, z.real))
            real = math.asinh(s1.real * s2.imag - s2.real * s1.imag)
            imag = math.atan2(z.imag, s1.real * s2.real - s1.imag * s2.imag)
            return complex(real, imag)
    return _asinh_impl


@overload(cmath.asin)
def asin_impl(z):
    if not isinstance(z, types.Complex):
        return

    def _asin_impl(z):
        """cmath.asin(z) = -j * cmath.asinh(z j)"""
        r = cmath.asinh(complex(-z.imag, z.real))
        return complex(r.imag, -r.real)
    return _asin_impl


@overload(cmath.atan)
def atan_impl(z):
    if not isinstance(z, types.Complex):
        return

    def _atan_impl(z):
        """cmath.atan(z) = -j * cmath.atanh(z j)"""
        r = cmath.atanh(complex(-z.imag, z.real))
        if math.isinf(z.real) and math.isnan(z.imag):
            # XXX this is odd but necessary
            return complex(r.imag, r.real)
        else:
            return complex(r.imag, -r.real)
    return _atan_impl


@overload(cmath.atanh)
def atanh_impl(z):
    if not isinstance(z, types.Complex):
        return

    THRES_LARGE = math.sqrt(mathimpl.FLT_MAX / 4)
    THRES_SMALL = math.sqrt(mathimpl.FLT_MIN)
    PI_12 = math.pi / 2

    def _atanh_impl(z):
        """cmath.atanh(z)"""
        # CPython's algorithm (see c_atanh() in cmathmodule.c)
        if z.real < 0.:
            # Reduce to case where z.real >= 0., using atanh(z) = -atanh(-z).
            negate = True
            z = -z
        else:
            negate = False

        ay = abs(z.imag)
        if math.isnan(z.real) or z.real > THRES_LARGE or ay > THRES_LARGE:
            if math.isinf(z.imag):
                real = math.copysign(0., z.real)
            elif math.isinf(z.real):
                real = 0.
            else:
                # may be safe from overflow, depending on hypot's implementation
                h = math.hypot(z.real * 0.5, z.imag * 0.5)
                real = z.real / 4. / h / h
            imag = -math.copysign(PI_12, -z.imag)
        elif z.real == 1. and ay < THRES_SMALL:
            # C99 standard says:  atanh(1+/-0.) should be inf +/- 0j
            if ay == 0.:
                real = INF
                imag = z.imag
            else:
                real = -math.log(math.sqrt(ay) /
                                 math.sqrt(math.hypot(ay, 2.)))
                imag = math.copysign(math.atan2(2., -ay) / 2, z.imag)
        else:
            sqay = ay * ay
            zr1 = 1 - z.real
            real = math.log1p(4. * z.real / (zr1 * zr1 + sqay)) * 0.25
            imag = -math.atan2(-2. * z.imag,
                               zr1 * (1 + z.real) - sqay) * 0.5

        if math.isnan(z.imag):
            imag = NAN
        if negate:
            return complex(-real, -imag)
        else:
            return complex(real, imag)

    return _atanh_impl
