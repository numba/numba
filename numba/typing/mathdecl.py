import math
from numba import types
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    builtin_global, builtin, signature)


@builtin
class MathModuleAttribute(AttributeTemplate):
    key = types.Module(math)

    def resolve_fabs(self, mod):
        return types.Function(Math_fabs)

    def resolve_exp(self, mod):
        return types.Function(Math_exp)

    def resolve_sqrt(self, mod):
        return types.Function(Math_sqrt)

    def resolve_log(self, mod):
        return types.Function(Math_log)

    def resolve_sin(self, mod):
        return types.Function(Math_sin)

    def resolve_cos(self, mod):
        return types.Function(Math_cos)

    def resolve_tan(self, mod):
        return types.Function(Math_tan)

    def resolve_sinh(self, mod):
        return types.Function(Math_sinh)

    def resolve_cosh(self, mod):
        return types.Function(Math_cosh)

    def resolve_tanh(self, mod):
        return types.Function(Math_tanh)

    def resolve_asin(self, mod):
        return types.Function(Math_asin)

    def resolve_acos(self, mod):
        return types.Function(Math_acos)

    def resolve_atan(self, mod):
        return types.Function(Math_atan)

    def resolve_asinh(self, mod):
        return types.Function(Math_asinh)

    def resolve_acosh(self, mod):
        return types.Function(Math_acosh)

    def resolve_atanh(self, mod):
        return types.Function(Math_atanh)

    def resolve_pi(self, mod):
        return types.float64

    def resolve_e(self, mod):
        return types.float64


class Math_unary(ConcreteTemplate):
    cases = [
        signature(types.float64, types.int64),
        signature(types.float64, types.uint64),
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
    ]


class Math_fabs(Math_unary):
    key = math.fabs


class Math_exp(Math_unary):
    key = math.exp


class Math_sqrt(Math_unary):
    key = math.sqrt


class Math_log(Math_unary):
    key = math.log


class Math_sin(Math_unary):
    key = math.sin


class Math_cos(Math_unary):
    key = math.cos


class Math_tan(Math_unary):
    key = math.tan


class Math_sinh(Math_unary):
    key = math.sinh


class Math_cosh(Math_unary):
    key = math.cosh


class Math_tanh(Math_unary):
    key = math.tanh


class Math_asin(Math_unary):
    key = math.asin


class Math_acos(Math_unary):
    key = math.acos


class Math_atan(Math_unary):
    key = math.atan


class Math_asinh(Math_unary):
    key = math.asinh


class Math_acosh(Math_unary):
    key = math.acosh


class Math_atanh(Math_unary):
    key = math.atanh


builtin_global(math, types.Module(math))
builtin_global(math.fabs, types.Function(Math_fabs))
builtin_global(math.exp, types.Function(Math_exp))
builtin_global(math.sqrt, types.Function(Math_sqrt))
builtin_global(math.log, types.Function(Math_log))
builtin_global(math.sin, types.Function(Math_sin))
builtin_global(math.cos, types.Function(Math_cos))
builtin_global(math.tan, types.Function(Math_tan))
builtin_global(math.sinh, types.Function(Math_sinh))
builtin_global(math.cosh, types.Function(Math_cosh))
builtin_global(math.tanh, types.Function(Math_tanh))
builtin_global(math.asin, types.Function(Math_asin))
builtin_global(math.acos, types.Function(Math_acos))
builtin_global(math.atan, types.Function(Math_atan))
builtin_global(math.asinh, types.Function(Math_asinh))
builtin_global(math.acosh, types.Function(Math_acosh))
builtin_global(math.atanh, types.Function(Math_atanh))
