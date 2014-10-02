"""
Provide math calls that uses intrinsics or libc math functions.
"""

from __future__ import print_function, absolute_import, division

import cmath

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


if utils.PYVERSION > (3, 2):
    @register
    @implement(cmath.isfinite, types.Kind(types.Complex))
    def isfinite_float_impl(context, builder, sig, args):
        [typ] = sig.args
        [value] = args
        cplx_cls = context.make_complex(typ)
        z = cplx_cls(context, builder, value=value)
        return is_finite(builder, z)

