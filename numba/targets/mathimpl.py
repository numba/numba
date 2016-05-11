"""
Provide math calls that uses intrinsics or libc math functions.
"""

from __future__ import print_function, absolute_import, division
import math
import sys

import llvmlite.llvmpy.core as lc
from llvmlite.llvmpy.core import Type

from numba.targets.imputils import Registry, impl_ret_untracked
from numba import types, cgutils, utils
from numba.typing import signature


registry = Registry()
lower = registry.lower


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
    # is_finite(x)  <=>  x - x != NaN
    val_minus_val = builder.fsub(val, val)
    return builder.fcmp_ordered('==', val_minus_val, val_minus_val)

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

def call_fp_intrinsic(builder, name, args):
    """
    Call a LLVM intrinsic floating-point operation.
    """
    mod = builder.module
    intr = lc.Function.intrinsic(mod, name, [a.type for a in args])
    return builder.call(intr, args)


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

def unary_math_int_impl(fn, float_impl):
    impl = _unary_int_input_wrapper_impl(float_impl)
    lower(fn, types.Integer)(impl)

def unary_math_intr(fn, intrcode):
    @lower(fn, types.Float)
    def float_impl(context, builder, sig, args):
        res = call_fp_intrinsic(builder, intrcode, args)
        return impl_ret_untracked(context, builder, sig.return_type, res)

    unary_math_int_impl(fn, float_impl)


def _float_input_unary_math_extern_impl(extern_func, input_type, restype=None):
    """
    Return an implementation factory to call unary *extern_func* with the
    given argument *input_type*.
    """
    def implementer(context, builder, sig, args):
        [val] = args
        mod = builder.module
        lty = context.get_value_type(input_type)
        fnty = Type.function(lty, [lty])
        fn = cgutils.insert_pure_function(builder.module, fnty, name=extern_func)
        res = builder.call(fn, (val,))
        if restype is not None:
            res = context.cast(builder, res, input_type, restype)
        return impl_ret_untracked(context, builder, sig.return_type, res)

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
    lower(fn, types.float32)(f32impl)
    lower(fn, types.float64)(f64impl)

    if int_restype:
        # If asked for an integral return type, we choose the input type
        # as the return type.
        for input_type in [types.intp, types.uintp, types.int64, types.uint64]:
            impl = _unary_int_input_wrapper_impl(
                _float_input_unary_math_extern_impl(f64extern, types.float64, input_type))
            lower(fn, input_type)(impl)
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
unary_math_extern(math.expm1, "expm1f", "expm1")
unary_math_extern(math.erf, "numba_erff", "numba_erf")
unary_math_extern(math.erfc, "numba_erfcf", "numba_erfc")
unary_math_extern(math.gamma, "numba_gammaf", "numba_gamma")
unary_math_extern(math.lgamma, "numba_lgammaf", "numba_lgamma")
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


@lower(math.isnan, types.Float)
def isnan_float_impl(context, builder, sig, args):
    [val] = args
    res = is_nan(builder, val)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower(math.isnan, types.Integer)
def isnan_int_impl(context, builder, sig, args):
    res = cgutils.false_bit
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower(math.isinf, types.Float)
def isinf_float_impl(context, builder, sig, args):
    [val] = args
    res = is_inf(builder, val)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower(math.isinf, types.Integer)
def isinf_int_impl(context, builder, sig, args):
    res = cgutils.false_bit
    return impl_ret_untracked(context, builder, sig.return_type, res)


if utils.PYVERSION >= (3, 2):
    @lower(math.isfinite, types.Float)
    def isfinite_float_impl(context, builder, sig, args):
        [val] = args
        res = is_finite(builder, val)
        return impl_ret_untracked(context, builder, sig.return_type, res)

    @lower(math.isfinite, types.Integer)
    def isfinite_int_impl(context, builder, sig, args):
        res = cgutils.true_bit
        return impl_ret_untracked(context, builder, sig.return_type, res)


@lower(math.copysign, types.Float, types.Float)
def copysign_float_impl(context, builder, sig, args):
    lty = args[0].type
    mod = builder.module
    fn = mod.get_or_insert_function(lc.Type.function(lty, (lty, lty)),
                                    'llvm.copysign.%s' % lty.intrinsic_name)
    res = builder.call(fn, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


# -----------------------------------------------------------------------------


@lower(math.frexp, types.Float)
def frexp_impl(context, builder, sig, args):
    val, = args
    fltty = context.get_data_type(sig.args[0])
    intty = context.get_data_type(sig.return_type[1])
    expptr = cgutils.alloca_once(builder, intty, name='exp')
    fnty = Type.function(fltty, (fltty, Type.pointer(intty)))
    fname = {
        "float": "numba_frexpf",
        "double": "numba_frexp",
        }[str(fltty)]
    fn = builder.module.get_or_insert_function(fnty, name=fname)
    res = builder.call(fn, (val, expptr))
    res = cgutils.make_anonymous_struct(builder, (res, builder.load(expptr)))
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower(math.ldexp, types.Float, types.intc)
def ldexp_impl(context, builder, sig, args):
    val, exp = args
    fltty, intty = map(context.get_data_type, sig.args)
    fnty = Type.function(fltty, (fltty, intty))
    fname = {
        "float": "numba_ldexpf",
        "double": "numba_ldexp",
        }[str(fltty)]
    fn = cgutils.insert_pure_function(builder.module, fnty, name=fname)
    res = builder.call(fn, (val, exp))
    return impl_ret_untracked(context, builder, sig.return_type, res)


# -----------------------------------------------------------------------------


@lower(math.atan2, types.int64, types.int64)
def atan2_s64_impl(context, builder, sig, args):
    [y, x] = args
    y = builder.sitofp(y, Type.double())
    x = builder.sitofp(x, Type.double())
    fsig = signature(types.float64, types.float64, types.float64)
    return atan2_f64_impl(context, builder, fsig, (y, x))

@lower(math.atan2, types.uint64, types.uint64)
def atan2_u64_impl(context, builder, sig, args):
    [y, x] = args
    y = builder.uitofp(y, Type.double())
    x = builder.uitofp(x, Type.double())
    fsig = signature(types.float64, types.float64, types.float64)
    return atan2_f64_impl(context, builder, fsig, (y, x))


@lower(math.atan2, types.float32, types.float32)
def atan2_f32_impl(context, builder, sig, args):
    assert len(args) == 2
    mod = builder.module
    fnty = Type.function(Type.float(), [Type.float(), Type.float()])
    fn = cgutils.insert_pure_function(builder.module, fnty, name="atan2f")
    res = builder.call(fn, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower(math.atan2, types.float64, types.float64)
def atan2_f64_impl(context, builder, sig, args):
    assert len(args) == 2
    mod = builder.module
    fnty = Type.function(Type.double(), [Type.double(), Type.double()])
    # Workaround atan2() issues under Windows
    fname = "atan2_fixed" if sys.platform == "win32" else "atan2"
    fn = cgutils.insert_pure_function(builder.module, fnty, name=fname)
    res = builder.call(fn, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


# -----------------------------------------------------------------------------


@lower(math.hypot, types.int64, types.int64)
def hypot_s64_impl(context, builder, sig, args):
    [x, y] = args
    y = builder.sitofp(y, Type.double())
    x = builder.sitofp(x, Type.double())
    fsig = signature(types.float64, types.float64, types.float64)
    res = hypot_float_impl(context, builder, fsig, (x, y))
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower(math.hypot, types.uint64, types.uint64)
def hypot_u64_impl(context, builder, sig, args):
    [x, y] = args
    y = builder.sitofp(y, Type.double())
    x = builder.sitofp(x, Type.double())
    fsig = signature(types.float64, types.float64, types.float64)
    res = hypot_float_impl(context, builder, fsig, (x, y))
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower(math.hypot, types.Float, types.Float)
def hypot_float_impl(context, builder, sig, args):
    def hypot(x, y):
        if math.isinf(x):
            return abs(x)
        elif math.isinf(y):
            return abs(y)
        return math.sqrt(x * x + y * y)

    res = context.compile_internal(builder, hypot, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


# -----------------------------------------------------------------------------

@lower(math.radians, types.Float)
def radians_float_impl(context, builder, sig, args):
    [x] = args
    coef = context.get_constant(sig.return_type, math.pi / 180)
    res = builder.fmul(x, coef)
    return impl_ret_untracked(context, builder, sig.return_type, res)

unary_math_int_impl(math.radians, radians_float_impl)

# -----------------------------------------------------------------------------

@lower(math.degrees, types.Float)
def degrees_float_impl(context, builder, sig, args):
    [x] = args
    coef = context.get_constant(sig.return_type, 180 / math.pi)
    res = builder.fmul(x, coef)
    return impl_ret_untracked(context, builder, sig.return_type, res)

unary_math_int_impl(math.degrees, degrees_float_impl)

# -----------------------------------------------------------------------------

@lower(math.pow, types.Float, types.Float)
@lower(math.pow, types.Float, types.Integer)
def pow_impl(context, builder, sig, args):
    impl = context.get_function("**", sig)
    return impl(builder, args)
