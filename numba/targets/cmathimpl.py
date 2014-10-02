"""
Provide math calls that uses intrinsics or libc math functions.
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
