import math
import numpy
from llvm.core import Type

from numbapro.npm.typing import (Conditional, Restrict, MustBe, cast_penalty,
                                 float_set, int_set, coerce, ScalarType)
from numbapro.npm.types import float32, float64

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

def binary_math(infer, call, obj):
    args = unpack_args(call)
    if len(args) != 2:
        raise TypeError("%s takes exactly two arguments" % obj)
    (op1, op2) = args
    infer.rules[call].add(Restrict(float_set))
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
}

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

}
