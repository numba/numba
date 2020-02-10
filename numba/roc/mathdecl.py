import math
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                         signature, Registry)

registry = Registry()
builtin_attr = registry.register_attr
infer_global = registry.register_global


@builtin_attr
class MathModuleAttribute(AttributeTemplate):
    key = types.Module(math)

    def resolve_fabs(self, mod):
        return types.Function(Math_fabs)

    def resolve_exp(self, mod):
        return types.Function(Math_exp)

    def resolve_expm1(self, mod):
        return types.Function(Math_expm1)

    def resolve_sqrt(self, mod):
        return types.Function(Math_sqrt)

    def resolve_log(self, mod):
        return types.Function(Math_log)

    def resolve_log1p(self, mod):
        return types.Function(Math_log1p)

    def resolve_log10(self, mod):
        return types.Function(Math_log10)

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

    def resolve_atan2(self, mod):
        return types.Function(Math_atan2)

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

    def resolve_floor(self, mod):
        return types.Function(Math_floor)

    def resolve_ceil(self, mod):
        return types.Function(Math_ceil)

    def resolve_trunc(self, mod):
        return types.Function(Math_trunc)

    def resolve_isnan(self, mod):
        return types.Function(Math_isnan)

    def resolve_isinf(self, mod):
        return types.Function(Math_isinf)

    def resolve_degrees(self, mod):
        return types.Function(Math_degrees)

    def resolve_radians(self, mod):
        return types.Function(Math_radians)

    # def resolve_hypot(self, mod):
    # return types.Function(Math_hypot)

    def resolve_copysign(self, mod):
        return types.Function(Math_copysign)

    def resolve_fmod(self, mod):
        return types.Function(Math_fmod)

    def resolve_pow(self, mod):
        return types.Function(Math_pow)

    def resolve_erf(self, mod):
        return types.Function(Math_erf)

    def resolve_erfc(self, mod):
        return types.Function(Math_erfc)

    def resolve_gamma(self, mod):
        return types.Function(Math_gamma)

    def resolve_lgamma(self, mod):
        return types.Function(Math_lgamma)


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


class Math_expm1(Math_unary):
    key = math.expm1


class Math_sqrt(Math_unary):
    key = math.sqrt


class Math_log(Math_unary):
    key = math.log


class Math_log1p(Math_unary):
    key = math.log1p


class Math_log10(Math_unary):
    key = math.log10


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


class Math_atan2(ConcreteTemplate):
    key = math.atan2
    cases = [
        signature(types.float64, types.int64, types.int64),
        signature(types.float64, types.uint64, types.uint64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]


class Math_asinh(Math_unary):
    key = math.asinh


class Math_acosh(Math_unary):
    key = math.acosh


class Math_atanh(Math_unary):
    key = math.atanh


class Math_floor(Math_unary):
    key = math.floor


class Math_ceil(Math_unary):
    key = math.ceil


class Math_trunc(Math_unary):
    key = math.trunc


class Math_radians(Math_unary):
    key = math.radians


class Math_degrees(Math_unary):
    key = math.degrees


# class Math_hypot(ConcreteTemplate):
# key = math.hypot
#     cases = [
#         signature(types.float64, types.int64, types.int64),
#         signature(types.float64, types.uint64, types.uint64),
#         signature(types.float32, types.float32, types.float32),
#         signature(types.float64, types.float64, types.float64),
#     ]


class Math_erf(Math_unary):
    key = math.erf

class Math_erfc(Math_unary):
    key = math.erfc

class Math_gamma(Math_unary):
    key = math.gamma

class Math_lgamma(Math_unary):
    key = math.lgamma


class Math_binary(ConcreteTemplate):
    cases = [
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]


class Math_copysign(Math_binary):
    key = math.copysign


class Math_fmod(Math_binary):
    key = math.fmod


class Math_pow(ConcreteTemplate):
    key = math.pow
    cases = [
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
        signature(types.float32, types.float32, types.int32),
        signature(types.float64, types.float64, types.int32),
    ]


class Math_isnan(ConcreteTemplate):
    key = math.isnan
    cases = [
        signature(types.boolean, types.int64),
        signature(types.boolean, types.uint64),
        signature(types.boolean, types.float32),
        signature(types.boolean, types.float64),
    ]


class Math_isinf(ConcreteTemplate):
    key = math.isinf
    cases = [
        signature(types.boolean, types.int64),
        signature(types.boolean, types.uint64),
        signature(types.boolean, types.float32),
        signature(types.boolean, types.float64),
    ]


infer_global(math, types.Module(math))
infer_global(math.fabs, types.Function(Math_fabs))
infer_global(math.exp, types.Function(Math_exp))
infer_global(math.expm1, types.Function(Math_expm1))
infer_global(math.sqrt, types.Function(Math_sqrt))
infer_global(math.log, types.Function(Math_log))
infer_global(math.log1p, types.Function(Math_log1p))
infer_global(math.log10, types.Function(Math_log10))
infer_global(math.sin, types.Function(Math_sin))
infer_global(math.cos, types.Function(Math_cos))
infer_global(math.tan, types.Function(Math_tan))
infer_global(math.sinh, types.Function(Math_sinh))
infer_global(math.cosh, types.Function(Math_cosh))
infer_global(math.tanh, types.Function(Math_tanh))
infer_global(math.asin, types.Function(Math_asin))
infer_global(math.acos, types.Function(Math_acos))
infer_global(math.atan, types.Function(Math_atan))
infer_global(math.atan2, types.Function(Math_atan2))
infer_global(math.asinh, types.Function(Math_asinh))
infer_global(math.acosh, types.Function(Math_acosh))
infer_global(math.atanh, types.Function(Math_atanh))
# infer_global(math.hypot, types.Function(Math_hypot))
infer_global(math.floor, types.Function(Math_floor))
infer_global(math.ceil, types.Function(Math_ceil))
infer_global(math.trunc, types.Function(Math_trunc))
infer_global(math.isnan, types.Function(Math_isnan))
infer_global(math.isinf, types.Function(Math_isinf))
infer_global(math.degrees, types.Function(Math_degrees))
infer_global(math.radians, types.Function(Math_radians))
infer_global(math.copysign, types.Function(Math_copysign))
infer_global(math.fmod, types.Function(Math_fmod))
infer_global(math.pow, types.Function(Math_pow))
infer_global(math.erf, types.Function(Math_erf))
infer_global(math.erfc, types.Function(Math_erfc))
infer_global(math.gamma, types.Function(Math_gamma))
infer_global(math.lgamma, types.Function(Math_lgamma))
