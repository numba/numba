from __future__ import print_function, absolute_import, division
import math

from numba.targets.imputils import implement, Registry
from numba import types
from .hsaimpl import _declare_function

registry = Registry()
register = registry.register

# -----------------------------------------------------------------------------


@register
@implement(math.sin, types.float32)
def math_sin(context, builder, sig, args):
    [val] = args
    fn = _declare_function(context, builder, 'sin', sig, ['float'])
    return builder.call(fn, [val])


@register
@implement(math.sin, types.float64)
def math_sin(context, builder, sig, args):
    [val] = args
    fn = _declare_function(context, builder, 'sin', sig, ['double'])
    return builder.call(fn, [val])


@register
@implement(math.cos, types.float32)
def math_cos(context, builder, sig, args):
    [val] = args
    fn = _declare_function(context, builder, 'cos', sig, ['float'])
    return builder.call(fn, [val])


@register
@implement(math.cos, types.float64)
def math_cos(context, builder, sig, args):
    [val] = args
    fn = _declare_function(context, builder, 'cos', sig, ['double'])
    return builder.call(fn, [val])

@register
@implement(math.tan, types.float32)
def math_tan(context, builder, sig, args):
    [val] = args
    fn = _declare_function(context, builder, 'tan', sig, ['float'])
    return builder.call(fn, [val])

@register
@implement(math.tan, types.float64)
def math_tan(context, builder, sig, args):
    [val] = args
    fn = _declare_function(context, builder, 'tan', sig, ['double'])
    return builder.call(fn, [val])
