import sys
import math
import numpy
from llvm.core import Type

from numbapro.npm.typing import (Conditional, Restrict, MustBe, cast_penalty,
                                 float_set, int_set, coerce, ScalarType)
from numbapro.npm.types import float32, float64, int32

#------------------------------------------------------------------------------
# Type Inference

def matches_unary(output, input):
    return output == input

def matches_binary(output, in1, in2):
    opty = coerce(in1, in2)
    return output == opty

def coerce_to_float(input):
    if input.is_float:
        return True
    elif input.is_int:
        output = ScalarType('f%d' % max(32, input.bitwidth))
        return cast_penalty(input, output)

def unpack_args(call):
    return [a.value for a in call.args.args]

def unary_math(infer, call, obj):
    args = unpack_args(call)
    if len(args) != 1:
        raise TypeError("%s takes exactly one argument" % obj)
    (operand,) = args
    infer.rules[call].add(Conditional(matches_unary, operand))
    infer.rules[call].add(Restrict(float_set))
    infer.rules[operand].add(Conditional(coerce_to_float))
    infer.rules[operand].add(Restrict(int_set|float_set))
    call.replace(func=obj) # normalize

def unary_bool_math(infer, call, obj):
    args = unpack_args(call)
    if len(args) != 1:
        raise TypeError("%s takes exactly one argument" % obj)
    (operand,) = args
    infer.rules[call].add(MustBe(int32))
    infer.rules[operand].add(Restrict(float_set))
    call.replace(func=obj) # normalize

def binary_math(infer, call, obj, integer=False):
    args = unpack_args(call)
    if len(args) != 2:
        raise TypeError("%s takes exactly two arguments" % obj)
    (op1, op2) = args
    if integer:
        rtset = int_set | float_set
    else:
        rtset = float_set

    infer.rules[call].add(Restrict(rtset))
    infer.rules[call].add(Conditional(matches_binary, op1, op2))
    infer.rules[op1].add(Restrict(int_set|float_set))
    infer.rules[op2].add(Restrict(int_set|float_set))
    infer.rules[op1].add(Conditional(coerce_to_float))
    infer.rules[op2].add(Conditional(coerce_to_float))
    call.replace(func=obj)

def rule_infer_acos(infer, call):
    unary_math(infer, call, math.acos)

def rule_infer_asin(infer, call):
    unary_math(infer, call, math.asin)

def rule_infer_atan(infer, call):
    unary_math(infer, call, math.atan)

def rule_infer_acosh(infer, call):
    unary_math(infer, call, math.acosh)

def rule_infer_asinh(infer, call):
    unary_math(infer, call, math.asinh)

def rule_infer_atanh(infer, call):
    unary_math(infer, call, math.atanh)


def rule_infer_cos(infer, call):
    unary_math(infer, call, math.cos)

def rule_infer_sin(infer, call):
    unary_math(infer, call, math.sin)

def rule_infer_tan(infer, call):
    unary_math(infer, call, math.tan)

def rule_infer_cosh(infer, call):
    unary_math(infer, call, math.cosh)

def rule_infer_sinh(infer, call):
    unary_math(infer, call, math.sinh)

def rule_infer_tanh(infer, call):
    unary_math(infer, call, math.tanh)


def rule_infer_atan2(infer, call):
    binary_math(infer, call, math.atan2)

def rule_infer_exp(infer, call):
    unary_math(infer, call, math.exp)

def rule_infer_expm1(infer, call):
    unary_math(infer, call, math.expm1)

def rule_infer_fabs(infer, call):
    unary_math(infer, call, math.fabs)

def rule_infer_log(infer, call):
    unary_math(infer, call, math.log)

def rule_infer_log10(infer, call):
    unary_math(infer, call, math.log10)

def rule_infer_log1p(infer, call):
    unary_math(infer, call, math.log1p)

def rule_infer_sqrt(infer, call):
    unary_math(infer, call, math.sqrt)

def rule_infer_pow(infer, call):
    binary_math(infer, call, math.pow, integer=True)

def rule_infer_ceil(infer, call):
    unary_math(infer, call, math.ceil)

def rule_infer_floor(infer, call):
    unary_math(infer, call, math.floor)

def rule_infer_copysign(infer, call):
    binary_math(infer, call, math.copysign)

def rule_infer_fmod(infer, call):
    binary_math(infer, call, math.fmod)

def rule_infer_isnan(infer, call):
    unary_bool_math(infer, call, math.isnan)

def rule_infer_isinf(infer, call):
    unary_bool_math(infer, call, math.isinf)

math_infer_rules = {
    # acos
    math.acos:           rule_infer_acos,
    numpy.arccos:        rule_infer_acos,
    # asin
    math.asin:           rule_infer_asin,
    numpy.arcsin:        rule_infer_asin,
    # atan
    math.atan:           rule_infer_atan,
    numpy.arctan:        rule_infer_atan,
    # acosh
    math.acosh:          rule_infer_acosh,
    numpy.arccosh:       rule_infer_acosh,
    # asinh
    math.asinh:           rule_infer_asinh,
    numpy.arcsinh:        rule_infer_asinh,
    # atan
    math.atanh:           rule_infer_atanh,
    numpy.arctanh:        rule_infer_atanh,

    # cos
    math.cos:               rule_infer_cos,
    numpy.cos:              rule_infer_cos,
    # sin
    math.sin:           	rule_infer_sin,
    numpy.sin:              rule_infer_sin,
    # tan
    math.tan:               rule_infer_tan,
    numpy.tan:              rule_infer_tan,
    # cosh
    math.cosh:              rule_infer_cosh,
    numpy.cosh:             rule_infer_cosh,
    # sinh
    math.sinh:              rule_infer_sinh,
    numpy.sinh:             rule_infer_sinh,
    # tan
    math.tanh:              rule_infer_tanh,
    numpy.tanh:             rule_infer_tanh,

    # atan2
    math.atan2:             rule_infer_atan2,
    numpy.arctan2:          rule_infer_atan2,

    # exp
    math.exp:               rule_infer_exp,
    numpy.exp:              rule_infer_exp,
    # expm1
    numpy.expm1:              rule_infer_expm1,
    # fabs
    math.fabs:               rule_infer_fabs,
    numpy.fabs:              rule_infer_fabs,
    # log
    math.log:               rule_infer_log,
    numpy.log:              rule_infer_log,
    # log10
    math.log10:               rule_infer_log10,
    numpy.log10:              rule_infer_log10,
    # log1p
    math.log1p:               rule_infer_log1p,
    numpy.log1p:              rule_infer_log1p,
    # sqrt
    math.sqrt:                rule_infer_sqrt,
    numpy.sqrt:               rule_infer_sqrt,

    # pow
    math.pow:                 rule_infer_pow,
    numpy.power:              rule_infer_pow,
    # ceil
    math.ceil:                rule_infer_ceil,
    numpy.ceil:               rule_infer_ceil,
    # floor
    math.floor:                rule_infer_floor,
    numpy.floor:               rule_infer_floor,
    # copysign
    math.copysign:              rule_infer_copysign,
    numpy.copysign:             rule_infer_copysign,
    # fmod
    math.fmod:              rule_infer_fmod,
    numpy.fmod:             rule_infer_fmod,
    # isnan
    math.isnan:             rule_infer_isnan,
    numpy.isnan:             rule_infer_isnan,
    # isinf
    math.isinf:             rule_infer_isinf,
    numpy.isinf:             rule_infer_isinf,
}

if sys.version_info[:2] >= (2, 7):
    math_infer_rules.update({
        math.expm1:               rule_infer_expm1,
    })

#------------------------------------------------------------------------------
# Code generation

def cg_math_unary(cg, call, disatchtable):
    restype = cg.typemap[call]
    (x,) = unpack_args(call)
    casted = cg.cast(x, restype)
    fname = disatchtable[restype]
    lrestype = cg.to_llvm(restype)
    func = cg.lmod.get_or_insert_function(Type.function(lrestype, [lrestype]),
                                          name=fname)
    res = cg.builder.call(func, [casted])
    cg.valmap[call] = res

def cg_math_bool_unary(cg, call, disatchtable):
    restype = cg.typemap[call]
    (x,) = unpack_args(call)
    argtype = cg.typemap[x]
    fname = disatchtable[argtype]
    lrestype = cg.to_llvm(restype)
    largtype = cg.to_llvm(argtype)
    func = cg.lmod.get_or_insert_function(Type.function(lrestype, [largtype]),
                                          name=fname)
    res = cg.builder.call(func, [cg.valmap[x]])
    cg.valmap[call] = res

def cg_math_binary(cg, call, disatchtable):
    restype = cg.typemap[call]
    (a, b) = unpack_args(call)
    ca = cg.cast(a, restype)
    cb = cg.cast(b, restype)
    fname = disatchtable[restype]
    ltype = cg.to_llvm(restype)
    func = cg.lmod.get_or_insert_function(Type.function(ltype, [ltype, ltype]),
                                          name=fname)
    res = cg.builder.call(func, [ca, cb])
    cg.valmap[call] = res

def cg_acos(cg, call):
    dispatch = {float64: '__nv_acos', float32: '__nv_acosf'}
    cg_math_unary(cg, call, dispatch)

def cg_asin(cg, call):
    dispatch = {float64: '__nv_asin', float32: '__nv_asinf'}
    cg_math_unary(cg, call, dispatch)

def cg_atan(cg, call):
    dispatch = {float64: '__nv_atan', float32: '__nv_atanf'}
    cg_math_unary(cg, call, dispatch)

def cg_acosh(cg, call):
    dispatch = {float64: '__nv_acosh', float32: '__nv_acoshf'}
    cg_math_unary(cg, call, dispatch)

def cg_asinh(cg, call):
    dispatch = {float64: '__nv_asinh', float32: '__nv_asinhf'}
    cg_math_unary(cg, call, dispatch)

def cg_atanh(cg, call):
    dispatch = {float64: '__nv_atanh', float32: '__nv_atanhf'}
    cg_math_unary(cg, call, dispatch)


def cg_cos(cg, call):
    dispatch = {float64: '__nv_cos', float32: '__nv_cosf'}
    cg_math_unary(cg, call, dispatch)

def cg_sin(cg, call):
    dispatch = {float64: '__nv_sin', float32: '__nv_sinf'}
    cg_math_unary(cg, call, dispatch)

def cg_tan(cg, call):
    dispatch = {float64: '__nv_tan', float32: '__nv_tanf'}
    cg_math_unary(cg, call, dispatch)

def cg_cosh(cg, call):
    dispatch = {float64: '__nv_cosh', float32: '__nv_coshf'}
    cg_math_unary(cg, call, dispatch)

def cg_sinh(cg, call):
    dispatch = {float64: '__nv_sinh', float32: '__nv_sinhf'}
    cg_math_unary(cg, call, dispatch)

def cg_tanh(cg, call):
    dispatch = {float64: '__nv_tanh', float32: '__nv_tanhf'}
    cg_math_unary(cg, call, dispatch)


def cg_atan2(cg, call):
    dispatch = {float64: '__nv_atan2', float32: '__nv_atan2f'}
    cg_math_binary(cg, call, dispatch)


def cg_exp(cg, call):
    dispatch = {float64: '__nv_exp', float32: '__nv_expf'}
    cg_math_unary(cg, call, dispatch)

def cg_expm1(cg, call):
    dispatch = {float64: '__nv_expm1', float32: '__nv_expm1f'}
    cg_math_unary(cg, call, dispatch)

def cg_fabs(cg, call):
    dispatch = {float64: '__nv_fabs', float32: '__nv_fabsf'}
    cg_math_unary(cg, call, dispatch)

def cg_log(cg, call):
    dispatch = {float64: '__nv_log', float32: '__nv_logf'}
    cg_math_unary(cg, call, dispatch)

def cg_log10(cg, call):
    dispatch = {float64: '__nv_log10', float32: '__nv_log10f'}
    cg_math_unary(cg, call, dispatch)

def cg_log1p(cg, call):
    dispatch = {float64: '__nv_log1p', float32: '__nv_log1pf'}
    cg_math_unary(cg, call, dispatch)


def cg_sqrt(cg, call):
    dispatch = {float64: '__nv_sqrt', float32: '__nv_sqrtf'}
    cg_math_unary(cg, call, dispatch)

def cg_pow(cg, call):
    dispatch = {float64: '__nv_pow', float32: '__nv_powf'}
    restype = cg.typemap[call]
    (a, b) = unpack_args(call)
    if cg.typemap[b].is_int:
        # handle integer power differently
        a = cg.cast(a, float64)
        b = cg.cast(b, int32)
        fname = '__nv_powi'
        types = [cg.to_llvm(t) for t in [float64, float64, int32]]
        lres, largs = types[0], types[1:]
        func = cg.lmod.get_or_insert_function(Type.function(lres, largs),
                                              name=fname)
        res = cg.builder.call(func, [a, b])
        cg.valmap[call] = cg.do_cast(res, float64, restype)
    else:
        cg_math_binary(cg, call, dispatch)

def cg_ceil(cg, call):
    dispatch = {float64: '__nv_ceil', float32: '__nv_ceilf'}
    cg_math_unary(cg, call, dispatch)

def cg_floor(cg, call):
    dispatch = {float64: '__nv_floor', float32: '__nv_floorf'}
    cg_math_unary(cg, call, dispatch)


def cg_copysign(cg, call):
    dispatch = {float64: '__nv_copysign', float32: '__nv_copysignf'}
    cg_math_binary(cg, call, dispatch)


def cg_fmod(cg, call):
    dispatch = {float64: '__nv_fmod', float32: '__nv_fmodf'}
    cg_math_binary(cg, call, dispatch)


def cg_isnan(cg, call):
    dispatch = {float64: '__nv_isnand', float32: '__nv_isnanf'}
    cg_math_bool_unary(cg, call, dispatch)

def cg_isinf(cg, call):
    dispatch = {float64: '__nv_isinfd', float32: '__nv_isinff'}
    cg_math_bool_unary(cg, call, dispatch)

math_codegen = {
    math.acos:          cg_acos,
    math.asin:          cg_asin,
    math.atan:          cg_atan,
    math.acosh:         cg_acosh,
    math.asinh:         cg_asinh,
    math.atanh:         cg_atanh,

    math.cos:           cg_cos,
    math.sin:           cg_sin,
    math.tan:           cg_tan,
    math.cosh:          cg_cosh,
    math.sinh:          cg_sinh,
    math.tanh:          cg_tanh,

    math.atan2:         cg_atan2,

    math.exp:           cg_exp,
    math.fabs:          cg_fabs,
    math.log:          cg_log,
    math.log10:          cg_log10,
    math.log1p:          cg_log1p,
    
    math.sqrt:          cg_sqrt,
    math.pow:           cg_pow,
    math.ceil:           cg_ceil,
    math.floor:           cg_floor,
    math.copysign:           cg_copysign,
    math.fmod:           cg_fmod,

    math.isnan:           cg_isnan,
    math.isinf:           cg_isinf,
}

if sys.version_info[:2] >= (2, 7):
    math_codegen.update({
        math.expm1:         cg_expm1,
    })


