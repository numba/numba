"""
Provide math calls that uses intrinsics or libc math functions.
"""

from __future__ import print_function, absolute_import, division
import math
import sys

import llvmlite.llvmpy.core as lc
from llvmlite.llvmpy.core import Type

from numba.targets.imputils import implement, Registry
from numba import types, cgutils, utils
from numba.typing import signature
from . import builtins


registry = Registry()
register = registry.register


# Helpers, shared with cmathimpl.

FLT_MAX = 3.402823466E+38
FLT_MIN = 1.175494351E-38

FLOAT_ABS_MASK = 0x7fffffff
FLOAT_SIGN_MASK = 0x80000000
DOUBLE_ABS_MASK = 0x7fffffffffffffff
DOUBLE_SIGN_MASK = 0x8000000000000000

def is_nan(builder, val):
    """
    Return a condition testing whether *val* is a NaN.
    """
    return builder.not_(builder.fcmp(lc.FCMP_OEQ, val, val))

def is_inf(builder, val):
    """
    Return a condition testing whether *val* is an infinite.
    """
    pos_inf = lc.Constant.real(val.type, float("+inf"))
    neg_inf = lc.Constant.real(val.type, float("-inf"))
    isposinf = builder.fcmp(lc.FCMP_OEQ, val, pos_inf)
    isneginf = builder.fcmp(lc.FCMP_OEQ, val, neg_inf)
    return builder.or_(isposinf, isneginf)

def is_finite(builder, val):
    """
    Return a condition testing whether *val* is a finite.
    """
    pos_inf = lc.Constant.real(val.type, float("+inf"))
    neg_inf = lc.Constant.real(val.type, float("-inf"))
    isnotposinf = builder.fcmp(lc.FCMP_ONE, val, pos_inf)
    isnotneginf = builder.fcmp(lc.FCMP_ONE, val, neg_inf)
    return builder.and_(isnotposinf, isnotneginf)

def f64_as_int64(builder, val):
    """
    Bitcast a double into a 64-bit integer.
    """
    assert val.type == Type.double()
    return builder.bitcast(val, Type.int(64))

def int64_as_f64(builder, val):
    """
    Bitcast a 64-bit integer into a double.
    """
    assert val.type == Type.int(64)
    return builder.bitcast(val, Type.double())

def f32_as_int32(builder, val):
    """
    Bitcast a float into a 32-bit integer.
    """
    assert val.type == Type.float()
    return builder.bitcast(val, Type.int(32))

def int32_as_f32(builder, val):
    """
    Bitcast a 32-bit integer into a float.
    """
    assert val.type == Type.int(32)
    return builder.bitcast(val, Type.float())

def negate_real(builder, val):
    """
    Negate real number *val*, with proper handling of zeros.
    """
    # The negative zero forces LLVM to handle signed zeros properly.
    return builder.fsub(lc.Constant.real(val.type, -0.0), val)


def _unary_int_input_wrapper_impl(wrapped_impl):
    """
    Return an implementation factory to convert the single integral input
    argument to a float, then defer to the *wrapped_impl*.
    """
    def implementer(context, builder, sig, args):
        [val] = args
        input_type = sig.args[0]
        if input_type.signed:
            fpval = builder.sitofp(val, Type.double())
        else:
            fpval = builder.uitofp(val, Type.double())
        sig = signature(types.float64, types.float64)
        return wrapped_impl(context, builder, sig, [fpval])

    return implementer

def unary_math_int_impl(fn, f64impl):
    impl = _unary_int_input_wrapper_impl(f64impl)
    for input_type in [types.intp, types.uintp, types.int64, types.uint64]:
        register(implement(fn, input_type)(impl))

def unary_math_intr(fn, intrcode):
    @register
    @implement(fn, types.float32)
    def f32impl(context, builder, sig, args):
        [val] = args
        mod = cgutils.get_module(builder)
        lty = context.get_value_type(types.float32)
        intr = lc.Function.intrinsic(mod, intrcode, [lty])
        return builder.call(intr, args)

    @register
    @implement(fn, types.float64)
    def f64impl(context, builder, sig, args):
        [val] = args
        mod = cgutils.get_module(builder)
        lty = context.get_value_type(types.float64)
        intr = lc.Function.intrinsic(mod, intrcode, [lty])
        return builder.call(intr, args)

    unary_math_int_impl(fn, f64impl)


def _float_input_unary_math_extern_impl(extern_func, input_type, restype=None):
    """
    Return an implementation factory to call unary *extern_func* with the
    given argument *input_type*.
    """
    def implementer(context, builder, sig, args):
        [val] = args
        mod = cgutils.get_module(builder)
        lty = context.get_value_type(input_type)
        fnty = Type.function(lty, [lty])
        fn = mod.get_or_insert_function(fnty, name=extern_func)
        res = builder.call(fn, (val,))
        if restype is None:
            return res
        else:
            return context.cast(builder, res, input_type,
                                restype)

    return implementer

def unary_math_extern(fn, f32extern, f64extern, int_restype=False):
    """
    Register implementations of Python function *fn* using the
    external function named *f32extern* and *f64extern* (for float32
    and float64 inputs, respectively).
    If *int_restype* is true, then the function's return value should be
    integral, otherwise floating-point.
    """
    f_restype = types.int64 if int_restype else None
    f32impl = _float_input_unary_math_extern_impl(f32extern, types.float32, f_restype)
    f64impl = _float_input_unary_math_extern_impl(f64extern, types.float64, f_restype)
    register(implement(fn, types.float32)(f32impl))
    register(implement(fn, types.float64)(f64impl))

    if int_restype:
        # If asked for an integral return type, we choose the input type
        # as the return type.
        for input_type in [types.intp, types.uintp, types.int64, types.uint64]:
            impl = _unary_int_input_wrapper_impl(
                _float_input_unary_math_extern_impl(f64extern, types.float64, input_type))
            register(implement(fn, input_type)(impl))
    else:
        unary_math_int_impl(fn, f64impl)


unary_math_intr(math.fabs, lc.INTR_FABS)
#unary_math_intr(math.sqrt, lc.INTR_SQRT)
unary_math_intr(math.exp, lc.INTR_EXP)
unary_math_intr(math.log, lc.INTR_LOG)
unary_math_intr(math.log10, lc.INTR_LOG10)
unary_math_intr(math.sin, lc.INTR_SIN)
unary_math_intr(math.cos, lc.INTR_COS)
#unary_math_intr(math.floor, lc.INTR_FLOOR)
#unary_math_intr(math.ceil, lc.INTR_CEIL)
#unary_math_intr(math.trunc, lc.INTR_TRUNC)
unary_math_extern(math.log1p, "log1pf", "log1p")
if utils.PYVERSION > (2, 6):
    unary_math_extern(math.expm1, "expm1f", "expm1")
unary_math_extern(math.tan, "tanf", "tan")
unary_math_extern(math.asin, "asinf", "asin")
unary_math_extern(math.acos, "acosf", "acos")
unary_math_extern(math.atan, "atanf", "atan")
unary_math_extern(math.asinh, "asinhf", "asinh")
unary_math_extern(math.acosh, "acoshf", "acosh")
unary_math_extern(math.atanh, "atanhf", "atanh")
unary_math_extern(math.sinh, "sinhf", "sinh")
unary_math_extern(math.cosh, "coshf", "cosh")
unary_math_extern(math.tanh, "tanhf", "tanh")
# math.floor and math.ceil return float on 2.x, int on 3.x
if utils.PYVERSION > (3, 0):
    unary_math_extern(math.ceil, "ceilf", "ceil", True)
    unary_math_extern(math.floor, "floorf", "floor", True)
else:
    unary_math_extern(math.ceil, "ceilf", "ceil")
    unary_math_extern(math.floor, "floorf", "floor")
unary_math_extern(math.sqrt, "sqrtf", "sqrt")
unary_math_extern(math.trunc, "truncf", "trunc", True)


@register
@implement(math.isnan, types.Kind(types.Float))
def isnan_float_impl(context, builder, sig, args):
    [val] = args
    return is_nan(builder, val)

@register
@implement(math.isnan, types.Kind(types.Integer))
def isnan_int_impl(context, builder, sig, args):
    return cgutils.false_bit


@register
@implement(math.isinf, types.Kind(types.Float))
def isinf_float_impl(context, builder, sig, args):
    [val] = args
    return is_inf(builder, val)

@register
@implement(math.isinf, types.Kind(types.Integer))
def isinf_int_impl(context, builder, sig, args):
    return cgutils.false_bit


if utils.PYVERSION >= (3, 2):
    @register
    @implement(math.isfinite, types.Kind(types.Float))
    def isfinite_float_impl(context, builder, sig, args):
        [val] = args
        return is_finite(builder, val)

    @register
    @implement(math.isfinite, types.Kind(types.Integer))
    def isfinite_int_impl(context, builder, sig, args):
        return cgutils.true_bit


# XXX copysign should use the corresponding LLVM intrinsic

@register
@implement(math.copysign, types.float32, types.float32)
def copysign_f32_impl(context, builder, sig, args):
    a = f32_as_int32(builder, args[0])
    b = f32_as_int32(builder, args[1])
    a = builder.and_(a, lc.Constant.int(a.type, FLOAT_ABS_MASK))
    b = builder.and_(b, lc.Constant.int(b.type, FLOAT_SIGN_MASK))
    res = builder.or_(a, b)
    return int32_as_f32(builder, res)

@register
@implement(math.copysign, types.float64, types.float64)
def copysign_f64_impl(context, builder, sig, args):
    a = f64_as_int64(builder, args[0])
    b = f64_as_int64(builder, args[1])
    a = builder.and_(a, lc.Constant.int(a.type, DOUBLE_ABS_MASK))
    b = builder.and_(b, lc.Constant.int(b.type, DOUBLE_SIGN_MASK))
    res = builder.or_(a, b)
    return int64_as_f64(builder, res)


# -----------------------------------------------------------------------------


@register
@implement(math.atan2, types.int64, types.int64)
def atan2_s64_impl(context, builder, sig, args):
    [y, x] = args
    y = builder.sitofp(y, Type.double())
    x = builder.sitofp(x, Type.double())
    fsig = signature(types.float64, types.float64, types.float64)
    return atan2_f64_impl(context, builder, fsig, (y, x))

@register
@implement(math.atan2, types.uint64, types.uint64)
def atan2_u64_impl(context, builder, sig, args):
    [y, x] = args
    y = builder.uitofp(y, Type.double())
    x = builder.uitofp(x, Type.double())
    fsig = signature(types.float64, types.float64, types.float64)
    return atan2_f64_impl(context, builder, fsig, (y, x))


@register
@implement(math.atan2, types.float32, types.float32)
def atan2_f32_impl(context, builder, sig, args):
    assert len(args) == 2
    mod = cgutils.get_module(builder)
    fnty = Type.function(Type.float(), [Type.float(), Type.float()])
    fn = mod.get_or_insert_function(fnty, name="atan2f")
    return builder.call(fn, args)

@register
@implement(math.atan2, types.float64, types.float64)
def atan2_f64_impl(context, builder, sig, args):
    assert len(args) == 2
    mod = cgutils.get_module(builder)
    fnty = Type.function(Type.double(), [Type.double(), Type.double()])
    # Workaround atan2() issues under Windows
    fname = "atan2_fixed" if sys.platform == "win32" else "atan2"
    fn = mod.get_or_insert_function(fnty, name=fname)
    return builder.call(fn, args)


# -----------------------------------------------------------------------------


@register
@implement(math.hypot, types.int64, types.int64)
def hypot_s64_impl(context, builder, sig, args):
    [x, y] = args
    y = builder.sitofp(y, Type.double())
    x = builder.sitofp(x, Type.double())
    fsig = signature(types.float64, types.float64, types.float64)
    return hypot_float_impl(context, builder, fsig, (x, y))

@register
@implement(math.hypot, types.uint64, types.uint64)
def hypot_u64_impl(context, builder, sig, args):
    [x, y] = args
    y = builder.sitofp(y, Type.double())
    x = builder.sitofp(x, Type.double())
    fsig = signature(types.float64, types.float64, types.float64)
    return hypot_float_impl(context, builder, fsig, (x, y))


@register
@implement(math.hypot, types.Kind(types.Float), types.Kind(types.Float))
def hypot_float_impl(context, builder, sig, args):
    def hypot(x, y):
        if math.isinf(x):
            return abs(x)
        elif math.isinf(y):
            return abs(y)
        return math.sqrt(x * x + y * y)

    return context.compile_internal(builder, hypot, sig, args)


# -----------------------------------------------------------------------------

@register
@implement(math.radians, types.float64)
def radians_f64_impl(context, builder, sig, args):
    [x] = args
    rate = builder.fdiv(x, context.get_constant(types.float64, 360))
    pi = context.get_constant(types.float64, math.pi)
    two = context.get_constant(types.float64, 2)
    twopi = builder.fmul(pi, two)
    return builder.fmul(rate, twopi)

@register
@implement(math.radians, types.float32)
def radians_f32_impl(context, builder, sig, args):
    [x] = args
    rate = builder.fdiv(x, context.get_constant(types.float32, 360))
    pi = context.get_constant(types.float32, math.pi)
    two = context.get_constant(types.float32, 2)
    twopi = builder.fmul(pi, two)
    return builder.fmul(rate, twopi)

unary_math_int_impl(math.radians, radians_f64_impl)

# -----------------------------------------------------------------------------

@register
@implement(math.degrees, types.float64)
def degrees_f64_impl(context, builder, sig, args):
    [x] = args
    full = context.get_constant(types.float64, 360)
    pi = context.get_constant(types.float64, math.pi)
    two = context.get_constant(types.float64, 2)
    twopi = builder.fmul(pi, two)
    return builder.fmul(builder.fdiv(x, twopi), full)

@register
@implement(math.degrees, types.float32)
def degrees_f32_impl(context, builder, sig, args):
    [x] = args
    full = context.get_constant(types.float32, 360)
    pi = context.get_constant(types.float32, math.pi)
    two = context.get_constant(types.float32, 2)
    twopi = builder.fmul(pi, two)
    return builder.fdiv(builder.fmul(x, full), twopi)

unary_math_int_impl(math.degrees, degrees_f64_impl)

# -----------------------------------------------------------------------------

for ty in types.unsigned_domain:
    register(implement(math.pow, types.float64, ty)(builtins.int_upower_impl))
for ty in types.signed_domain:
    register(implement(math.pow, types.float64, ty)(builtins.int_spower_impl))
for ty in types.real_domain:
    register(implement(math.pow, ty, ty)(builtins.real_power_impl))

