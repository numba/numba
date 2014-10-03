"""
Implement the cmath module functions.
"""

from __future__ import print_function, absolute_import, division

import cmath
import math

import llvm.core as lc
from llvm.core import Type

from numba.targets.imputils import implement, Registry
from numba import types, cgutils, utils
from numba.typing import signature
from . import builtins, mathimpl


registry = Registry()
register = registry.register


def is_nan(builder, z):
    return builder.or_(mathimpl.is_nan(builder, z.real),
                       mathimpl.is_nan(builder, z.imag))

def is_inf(builder, z):
    return builder.or_(mathimpl.is_inf(builder, z.real),
                       mathimpl.is_inf(builder, z.imag))

def is_finite(builder, z):
    return builder.and_(mathimpl.is_finite(builder, z.real),
                        mathimpl.is_finite(builder, z.imag))


@register
@implement(cmath.isnan, types.Kind(types.Complex))
def isnan_float_impl(context, builder, sig, args):
    [typ] = sig.args
    [value] = args
    cplx_cls = context.make_complex(typ)
    z = cplx_cls(context, builder, value=value)
    return is_nan(builder, z)

@register
@implement(cmath.isinf, types.Kind(types.Complex))
def isinf_float_impl(context, builder, sig, args):
    [typ] = sig.args
    [value] = args
    cplx_cls = context.make_complex(typ)
    z = cplx_cls(context, builder, value=value)
    return is_inf(builder, z)


if utils.PYVERSION >= (3, 2):
    @register
    @implement(cmath.isfinite, types.Kind(types.Complex))
    def isfinite_float_impl(context, builder, sig, args):
        [typ] = sig.args
        [value] = args
        cplx_cls = context.make_complex(typ)
        z = cplx_cls(context, builder, value=value)
        return is_finite(builder, z)


@register
@implement(cmath.rect, types.Kind(types.Float), types.Kind(types.Float))
def rect_impl(context, builder, sig, args):
    [r, phi] = args
    # We can't call math.isfinite() inside rect() below because it
    # only exists on 3.2+.
    phi_is_finite = mathimpl.is_finite(builder, phi)

    def rect(r, phi, phi_is_finite):
        if not phi_is_finite:
            if not r:
                # cmath.rect(0, phi={inf, nan}) = 0
                return complex(r, r)
            if math.isinf(r):
                # cmath.rect(inf, phi={inf, nan}) = inf + j phi
                return complex(r, phi)
        if not phi:
            # cmath.rect(r, 0) = r
            return complex(r, phi)
        return r * complex(math.cos(phi), math.sin(phi))

    inner_sig = signature(sig.return_type, *sig.args + (types.boolean,))
    return context.compile_internal(builder, rect, inner_sig,
                                    args + [phi_is_finite])


def intrinsic_complex_unary(inner_func):
    def wrapper(context, builder, sig, args):
        [typ] = sig.args
        [value] = args
        cplx_cls = context.make_complex(typ)
        z = cplx_cls(context, builder, value=value)
        x = z.real
        y = z.imag
        # Same as above: math.isfinite() is unavailable on 2.x so we precompute
        # its value and pass it to the pure Python implementation.
        x_is_finite = mathimpl.is_finite(builder, x)
        y_is_finite = mathimpl.is_finite(builder, y)
        inner_sig = signature(sig.return_type,
                              *(typ.underlying_float,) * 2 + (types.boolean,) * 2)
        return context.compile_internal(builder, inner_func, inner_sig,
                                        (x, y, x_is_finite, y_is_finite))
    return wrapper


NAN = float('nan')

@register
@implement(cmath.exp, types.Kind(types.Complex))
@intrinsic_complex_unary
def exp_impl(x, y, x_is_finite, y_is_finite):
    """cmath.exp(x + y j)"""
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
            c = math.cos(y)
            s = math.sin(y)
            return complex(x * c, y * s)
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
            return complex(r, r)

@register
@implement(cmath.log, types.Kind(types.Complex))
@intrinsic_complex_unary
def log_impl(x, y, x_is_finite, y_is_finite):
    """cmath.log(x + y j)"""
    a = math.log(math.hypot(x, y))
    b = math.atan2(y, x)
    return complex(a, b)


@register
@implement(cmath.log, types.Kind(types.Complex), types.Kind(types.Complex))
def log_base_impl(context, builder, sig, args):
    """cmath.log(z, base)"""
    [z, base] = args

    def log_base(z, base):
        return cmath.log(z) / cmath.log(base)

    return context.compile_internal(builder, log_base, sig, args)


@register
@implement(cmath.phase, types.Kind(types.Complex))
@intrinsic_complex_unary
def phase_impl(x, y, x_is_finite, y_is_finite):
    """cmath.phase(x + y j)"""
    return math.atan2(y, x)

@register
@implement(cmath.polar, types.Kind(types.Complex))
@intrinsic_complex_unary
def polar_impl(x, y, x_is_finite, y_is_finite):
    """cmath.polar(x + y j)"""
    return math.hypot(x, y), math.atan2(y, x)
