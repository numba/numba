import math
from numba import types, utils
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    signature, Registry)

registry = Registry()
builtin_attr = registry.register_attr
builtin_global = registry.register_global


class Math_unary(ConcreteTemplate):
    cases = [
        signature(types.float64, types.int64),
        signature(types.float64, types.uint64),
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
    ]


class Math_converter(ConcreteTemplate):
    cases = [
        signature(types.intp, types.intp),
        signature(types.int64, types.int64),
        signature(types.uint64, types.uint64),
        signature(types.int64, types.float32),
        signature(types.int64, types.float64),
    ]


class Math_fabs(Math_unary):
    key = math.fabs


class Math_exp(Math_unary):
    key = math.exp


if utils.PYVERSION > (2, 6):
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



# math.floor and math.ceil return float on 2.x, int on 3.x
if utils.PYVERSION > (3, 0):

    class Math_floor(Math_converter):
        key = math.floor


    class Math_ceil(Math_converter):
        key = math.ceil


else:

    class Math_floor(Math_unary):
        key = math.floor


    class Math_ceil(Math_unary):
        key = math.ceil


class Math_trunc(Math_converter):
    key = math.trunc


class Math_radians(Math_unary):
    key = math.radians


class Math_degrees(Math_unary):
    key = math.degrees


class Math_hypot(ConcreteTemplate):
    key = math.hypot
    cases = [
        signature(types.float64, types.int64, types.int64),
        signature(types.float64, types.uint64, types.uint64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
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


class Math_pow(ConcreteTemplate):
    key = math.pow
    cases = [
        signature(types.float64, types.float64, types.int64),
        signature(types.float64, types.float64, types.uint64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]


builtin_global(math, types.Module(math))
builtin_global(math.fabs, types.Function(Math_fabs))
builtin_global(math.exp, types.Function(Math_exp))
if utils.PYVERSION > (2, 6):
    builtin_global(math.expm1, types.Function(Math_expm1))
builtin_global(math.sqrt, types.Function(Math_sqrt))
builtin_global(math.log, types.Function(Math_log))
builtin_global(math.log1p, types.Function(Math_log1p))
builtin_global(math.log10, types.Function(Math_log10))
builtin_global(math.sin, types.Function(Math_sin))
builtin_global(math.cos, types.Function(Math_cos))
builtin_global(math.tan, types.Function(Math_tan))
builtin_global(math.sinh, types.Function(Math_sinh))
builtin_global(math.cosh, types.Function(Math_cosh))
builtin_global(math.tanh, types.Function(Math_tanh))
builtin_global(math.asin, types.Function(Math_asin))
builtin_global(math.acos, types.Function(Math_acos))
builtin_global(math.atan, types.Function(Math_atan))
builtin_global(math.atan2, types.Function(Math_atan2))
builtin_global(math.asinh, types.Function(Math_asinh))
builtin_global(math.acosh, types.Function(Math_acosh))
builtin_global(math.atanh, types.Function(Math_atanh))
builtin_global(math.hypot, types.Function(Math_hypot))
builtin_global(math.floor, types.Function(Math_floor))
builtin_global(math.ceil, types.Function(Math_ceil))
builtin_global(math.trunc, types.Function(Math_trunc))
builtin_global(math.isnan, types.Function(Math_isnan))
builtin_global(math.isinf, types.Function(Math_isinf))
builtin_global(math.degrees, types.Function(Math_degrees))
builtin_global(math.radians, types.Function(Math_radians))
builtin_global(math.pow, types.Function(Math_pow))
