"""
Provide math calls that uses intrinsics or libc math functions.
"""

from __future__ import print_function, absolute_import, division
import math
import llvm.core as lc
from llvm.core import Type
from numba.targets.imputils import implement, impl_attribute, builtin_attr
from numba import types, cgutils


functions = []


def register(f):
    functions.append(f)
    return f


def unary_math_int_impl(fn, f64impl):
    @register
    @implement(fn, types.int64)
    def s64impl(context, builder, sig, args):
        [val] = args
        fpval = builder.sitofp(val, Type.double())
        return f64impl(context, builder, [types.float64, types.float64],
                       [fpval])

    @register
    @implement(fn, types.uint64)
    def u64impl(context, builder, sig, args):
        [val] = args
        fpval = builder.uitofp(val, Type.double())
        return f64impl(context, builder, [types.float64, types.float64],
                       [fpval])


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


def unary_math_extern(fn, f32extern, f64extern):
    @register
    @implement(fn, types.float32)
    def f32impl(context, builder, sig, args):
        [val] = args
        mod = cgutils.get_module(builder)
        fnty = Type.function(Type.float(), [Type.float()])
        fn = mod.get_or_insert_function(fnty, name=f32extern)
        return builder.call(fn, (val,))

    @register
    @implement(fn, types.float64)
    def f64impl(context, builder, sig, args):
        [val] = args
        mod = cgutils.get_module(builder)
        fnty = Type.function(Type.double(), [Type.double()])
        fn = mod.get_or_insert_function(fnty, name=f64extern)
        return builder.call(fn, (val,))

    unary_math_int_impl(fn, f64impl)


unary_math_intr(math.fabs, lc.INTR_FABS)
unary_math_intr(math.sqrt, lc.INTR_SQRT)
unary_math_intr(math.exp, lc.INTR_EXP)
unary_math_intr(math.log, lc.INTR_LOG)
unary_math_intr(math.sin, lc.INTR_SIN)
unary_math_intr(math.cos, lc.INTR_COS)
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


@builtin_attr
@impl_attribute(types.Module(math), "pi", types.float64)
def math_pi_impl(context, builder, typ, value):
    return context.get_constant(types.float64, math.pi)

@builtin_attr
@impl_attribute(types.Module(math), "e", types.float64)
def math_e_impl(context, builder, typ, value):
    return context.get_constant(types.float64, math.e)
