"""
Provide math calls that uses intrinsics or libc math functions.
"""

from __future__ import print_function, absolute_import, division
import math
import operator
import sys
import numpy as np

import llvmlite.llvmpy.core as lc
from llvmlite.llvmpy.core import Type

from numba.targets.imputils import Registry, impl_ret_untracked
from numba import types, cgutils, utils, config
from numba.typing import signature


registry = Registry()
lower = registry.lower


# Helpers, shared with cmathimpl.
_NP_FLT_FINFO = np.finfo(np.dtype('float32'))
FLT_MAX = _NP_FLT_FINFO.max
FLT_MIN = _NP_FLT_FINFO.tiny

_NP_DBL_FINFO = np.finfo(np.dtype('float64'))
DBL_MAX = _NP_DBL_FINFO.max
DBL_MIN = _NP_DBL_FINFO.tiny

FLOAT_ABS_MASK = 0x7fffffff
FLOAT_SIGN_MASK = 0x80000000
DOUBLE_ABS_MASK = 0x7fffffffffffffff
DOUBLE_SIGN_MASK = 0x8000000000000000


def is_nan(builder, val):
    """
    Return a condition testing whether *val* is a NaN.
    """
    return builder.fcmp_unordered('uno', val, val)

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
    return builder.fcmp_ordered('ord', val_minus_val, val_minus_val)

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
    argument to a float64, then defer to the *wrapped_impl*.
    """
    def implementer(context, builder, sig, args):
        val, = args
        input_type = sig.args[0]
        fpval = context.cast(builder, val, input_type, types.float64)
        inner_sig = signature(types.float64, types.float64)
        res = wrapped_impl(context, builder, inner_sig, (fpval,))
        return context.cast(builder, res, types.float64, sig.return_type)

    return implementer

def unary_math_int_impl(fn, float_impl):
    impl = _unary_int_input_wrapper_impl(float_impl)
    lower(fn, types.Integer)(impl)

def unary_math_intr(fn, intrcode):
    """
    Implement the math function *fn* using the LLVM intrinsic *intrcode*.
    """
    @lower(fn, types.Float)
    def float_impl(context, builder, sig, args):
        res = call_fp_intrinsic(builder, intrcode, args)
        return impl_ret_untracked(context, builder, sig.return_type, res)

    unary_math_int_impl(fn, float_impl)
    return float_impl

def unary_math_extern(fn, f32extern, f64extern, int_restype=False):
    """
    Register implementations of Python function *fn* using the
    external function named *f32extern* and *f64extern* (for float32
    and float64 inputs, respectively).
    If *int_restype* is true, then the function's return value should be
    integral, otherwise floating-point.
    """
    f_restype = types.int64 if int_restype else None

    def float_impl(context, builder, sig, args):
        """
        Implement *fn* for a types.Float input.
        """
        [val] = args
        mod = builder.module
        input_type = sig.args[0]
        lty = context.get_value_type(input_type)
        func_name = {
            types.float32: f32extern,
            types.float64: f64extern,
            }[input_type]
        fnty = Type.function(lty, [lty])
        fn = cgutils.insert_pure_function(builder.module, fnty, name=func_name)
        res = builder.call(fn, (val,))
        res = context.cast(builder, res, input_type, sig.return_type)
        return impl_ret_untracked(context, builder, sig.return_type, res)

    lower(fn, types.Float)(float_impl)

    # Implement wrapper for integer inputs
    unary_math_int_impl(fn, float_impl)

    return float_impl


unary_math_intr(math.fabs, lc.INTR_FABS)
#unary_math_intr(math.sqrt, lc.INTR_SQRT)
exp_impl = unary_math_intr(math.exp, lc.INTR_EXP)
log_impl = unary_math_intr(math.log, lc.INTR_LOG)
log10_impl = unary_math_intr(math.log10, lc.INTR_LOG10)
sin_impl = unary_math_intr(math.sin, lc.INTR_SIN)
cos_impl = unary_math_intr(math.cos, lc.INTR_COS)
#unary_math_intr(math.floor, lc.INTR_FLOOR)
#unary_math_intr(math.ceil, lc.INTR_CEIL)
#unary_math_intr(math.trunc, lc.INTR_TRUNC)

log1p_impl = unary_math_extern(math.log1p, "log1pf", "log1p")
expm1_impl = unary_math_extern(math.expm1, "expm1f", "expm1")
erf_impl = unary_math_extern(math.erf, "erff", "erf")
erfc_impl = unary_math_extern(math.erfc, "erfcf", "erfc")

tan_impl = unary_math_extern(math.tan, "tanf", "tan")
asin_impl = unary_math_extern(math.asin, "asinf", "asin")
acos_impl = unary_math_extern(math.acos, "acosf", "acos")
atan_impl = unary_math_extern(math.atan, "atanf", "atan")

asinh_impl = unary_math_extern(math.asinh, "asinhf", "asinh")
acosh_impl = unary_math_extern(math.acosh, "acoshf", "acosh")
atanh_impl = unary_math_extern(math.atanh, "atanhf", "atanh")
sinh_impl = unary_math_extern(math.sinh, "sinhf", "sinh")
cosh_impl = unary_math_extern(math.cosh, "coshf", "cosh")
tanh_impl = unary_math_extern(math.tanh, "tanhf", "tanh")

# math.floor and math.ceil return float on 2.x, int on 3.x
if utils.PYVERSION > (3, 0):
    log2_impl = unary_math_extern(math.log2, "log2f", "log2")
    ceil_impl = unary_math_extern(math.ceil, "ceilf", "ceil", True)
    floor_impl = unary_math_extern(math.floor, "floorf", "floor", True)
else:
    ceil_impl = unary_math_extern(math.ceil, "ceilf", "ceil")
    floor_impl = unary_math_extern(math.floor, "floorf", "floor")
gamma_impl = unary_math_extern(math.gamma, "numba_gammaf", "numba_gamma") # work-around
sqrt_impl = unary_math_extern(math.sqrt, "sqrtf", "sqrt")
trunc_impl = unary_math_extern(math.trunc, "truncf", "trunc", True)
lgamma_impl = unary_math_extern(math.lgamma, "lgammaf", "lgamma")


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
    return atan2_float_impl(context, builder, fsig, (y, x))

@lower(math.atan2, types.uint64, types.uint64)
def atan2_u64_impl(context, builder, sig, args):
    [y, x] = args
    y = builder.uitofp(y, Type.double())
    x = builder.uitofp(x, Type.double())
    fsig = signature(types.float64, types.float64, types.float64)
    return atan2_float_impl(context, builder, fsig, (y, x))

@lower(math.atan2, types.Float, types.Float)
def atan2_float_impl(context, builder, sig, args):
    assert len(args) == 2
    mod = builder.module
    ty = sig.args[0]
    lty = context.get_value_type(ty)
    func_name = {
        types.float32: "atan2f",
        # Workaround atan2() issues under Windows
        types.float64: "atan2_fixed" if sys.platform == "win32" else "atan2"
        }[ty]
    fnty = Type.function(lty, (lty, lty))
    fn = cgutils.insert_pure_function(builder.module, fnty, name=func_name)
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
    xty, yty = sig.args
    assert xty == yty == sig.return_type
    x, y = args

    # Windows has alternate names for hypot/hypotf, see
    # https://msdn.microsoft.com/fr-fr/library/a9yb3dbt%28v=vs.80%29.aspx
    fname = {
        types.float32: "_hypotf" if sys.platform == 'win32' else "hypotf",
        types.float64: "_hypot" if sys.platform == 'win32' else "hypot",
    }[xty]
    plat_hypot = types.ExternalFunction(fname, sig)

    if sys.platform == 'win32' and config.MACHINE_BITS == 32:
        inf = xty(float('inf'))

        def hypot_impl(x, y):
            if math.isinf(x) or math.isinf(y):
                return inf
            return plat_hypot(x, y)
    else:
        def hypot_impl(x, y):
            return plat_hypot(x, y)

    res = context.compile_internal(builder, hypot_impl, sig, args)
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
    impl = context.get_function(operator.pow, sig)
    return impl(builder, args)
