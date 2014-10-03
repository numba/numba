import math
from numba import types, utils
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    signature, Registry)

registry = Registry()
builtin_global = registry.register_global


@registry.resolves_global(math.exp)
@registry.resolves_global(math.fabs)
@registry.resolves_global(math.sqrt)
@registry.resolves_global(math.log)
@registry.resolves_global(math.log1p)
@registry.resolves_global(math.log10)
@registry.resolves_global(math.sin)
@registry.resolves_global(math.cos)
@registry.resolves_global(math.tan)
@registry.resolves_global(math.sinh)
@registry.resolves_global(math.cosh)
@registry.resolves_global(math.tanh)
@registry.resolves_global(math.asin)
@registry.resolves_global(math.acos)
@registry.resolves_global(math.atan)
@registry.resolves_global(math.asinh)
@registry.resolves_global(math.acosh)
@registry.resolves_global(math.atanh)
@registry.resolves_global(math.degrees)
@registry.resolves_global(math.radians)
class Math_unary(ConcreteTemplate):
    cases = [
        signature(types.float64, types.int64),
        signature(types.float64, types.uint64),
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
    ]

@registry.resolves_global(math.atan2)
class Math_atan2(ConcreteTemplate):
    cases = [
        signature(types.float64, types.int64, types.int64),
        signature(types.float64, types.uint64, types.uint64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]

if utils.PYVERSION > (2, 6):
    @registry.resolves_global(math.expm1)
    class Math_expm1(Math_unary):
        pass

@registry.resolves_global(math.trunc)
class Math_converter(ConcreteTemplate):
    cases = [
        signature(types.intp, types.intp),
        signature(types.int64, types.int64),
        signature(types.uint64, types.uint64),
        signature(types.int64, types.float32),
        signature(types.int64, types.float64),
    ]

# math.floor and math.ceil return float on 2.x, int on 3.x
if utils.PYVERSION > (3, 0):
    @registry.resolves_global(math.floor)
    @registry.resolves_global(math.ceil)
    class Math_floor_ceil(Math_converter):
        pass
else:
    @registry.resolves_global(math.floor)
    @registry.resolves_global(math.ceil)
    class Math_floor_ceil(Math_unary):
        pass


@registry.resolves_global(math.copysign)
class Math_copysign(ConcreteTemplate):
    cases = [
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]


@registry.resolves_global(math.hypot)
class Math_hypot(ConcreteTemplate):
    cases = [
        signature(types.float64, types.int64, types.int64),
        signature(types.float64, types.uint64, types.uint64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]


@registry.resolves_global(math.isinf)
@registry.resolves_global(math.isnan)
class Math_predicate(ConcreteTemplate):
    cases = [
        signature(types.boolean, types.int64),
        signature(types.boolean, types.uint64),
        signature(types.boolean, types.float32),
        signature(types.boolean, types.float64),
    ]

if utils.PYVERSION >= (3, 2):
    @registry.resolves_global(math.isfinite)
    class Math_isfinite(Math_predicate):
        pass


@registry.resolves_global(math.pow)
class Math_pow(ConcreteTemplate):
    cases = [
        signature(types.float64, types.float64, types.int64),
        signature(types.float64, types.float64, types.uint64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]


builtin_global(math, types.Module(math))
