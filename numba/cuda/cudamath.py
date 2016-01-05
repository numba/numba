from __future__ import print_function, absolute_import, division
import math
from numba import types, utils
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    signature, Registry)

registry = Registry()
builtin_attr = registry.register_attr
builtin_global = registry.register_global


@builtin_global(math.acos)
@builtin_global(math.acosh)
@builtin_global(math.asin)
@builtin_global(math.asinh)
@builtin_global(math.atan)
@builtin_global(math.atanh)
@builtin_global(math.ceil)
@builtin_global(math.cos)
@builtin_global(math.cosh)
@builtin_global(math.degrees)
@builtin_global(math.exp)
@builtin_global(math.fabs)
@builtin_global(math.floor)
@builtin_global(math.log)
@builtin_global(math.log10)
@builtin_global(math.log1p)
@builtin_global(math.radians)
@builtin_global(math.sin)
@builtin_global(math.sinh)
@builtin_global(math.sqrt)
@builtin_global(math.tan)
@builtin_global(math.tanh)
@builtin_global(math.trunc)
class Math_unary(ConcreteTemplate):
    cases = [
        signature(types.float64, types.int64),
        signature(types.float64, types.uint64),
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
    ]


if utils.PYVERSION > (2, 6):
    builtin_global(math.expm1)(Math_unary)


@builtin_global(math.atan2)
class Math_atan2(ConcreteTemplate):
    key = math.atan2
    cases = [
        signature(types.float64, types.int64, types.int64),
        signature(types.float64, types.uint64, types.uint64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]


@builtin_global(math.hypot)
class Math_hypot(ConcreteTemplate):
    key = math.hypot
    cases = [
        signature(types.float64, types.int64, types.int64),
        signature(types.float64, types.uint64, types.uint64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]


@builtin_global(math.copysign)
@builtin_global(math.fmod)
class Math_binary(ConcreteTemplate):
    cases = [
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]


@builtin_global(math.pow)
class Math_pow(ConcreteTemplate):
    cases = [
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
        signature(types.float32, types.float32, types.int32),
        signature(types.float64, types.float64, types.int32),
    ]


@builtin_global(math.isinf)
@builtin_global(math.isnan)
class Math_isnan(ConcreteTemplate):
    cases = [
        signature(types.boolean, types.int64),
        signature(types.boolean, types.uint64),
        signature(types.boolean, types.float32),
        signature(types.boolean, types.float64),
    ]
