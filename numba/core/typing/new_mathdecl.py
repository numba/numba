import math
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                         signature, Registry)

registry = Registry()
infer_global = registry.register_global


@infer_global(math.exp)
@infer_global(math.expm1)
@infer_global(math.fabs)
@infer_global(math.sqrt)
@infer_global(math.log)
@infer_global(math.log1p)
@infer_global(math.log10)
@infer_global(math.log2)
@infer_global(math.sin)
@infer_global(math.cos)
@infer_global(math.tan)
@infer_global(math.sinh)
@infer_global(math.cosh)
@infer_global(math.tanh)
@infer_global(math.asin)
@infer_global(math.acos)
@infer_global(math.atan)
@infer_global(math.asinh)
@infer_global(math.acosh)
@infer_global(math.atanh)
@infer_global(math.degrees)
@infer_global(math.radians)
@infer_global(math.erf)
@infer_global(math.erfc)
@infer_global(math.gamma)
@infer_global(math.lgamma)
class Math_unary(ConcreteTemplate):
    cases = [
        signature(types.np_float64, types.np_int64),
        signature(types.np_float64, types.np_uint64),
        signature(types.np_float32, types.np_float32),
        signature(types.np_float64, types.np_float64),
    ]


@infer_global(math.atan2)
class Math_atan2(ConcreteTemplate):
    cases = [
        signature(types.np_float64, types.np_int64, types.np_int64),
        signature(types.np_float64, types.np_uint64, types.np_uint64),
        signature(types.np_float32, types.np_float32, types.np_float32),
        signature(types.np_float64, types.np_float64, types.np_float64),
    ]


@infer_global(math.trunc)
class Math_converter(ConcreteTemplate):
    cases = [
        signature(types.np_intp, types.np_intp),
        signature(types.np_int64, types.np_int64),
        signature(types.np_uint64, types.np_uint64),
        signature(types.np_int64, types.np_float32),
        signature(types.np_int64, types.np_float64),
    ]


@infer_global(math.floor)
@infer_global(math.ceil)
class Math_floor_ceil(Math_converter):
    pass


@infer_global(math.copysign)
class Math_copysign(ConcreteTemplate):
    cases = [
        signature(types.np_float32, types.np_float32, types.np_float32),
        signature(types.np_float64, types.np_float64, types.np_float64),
    ]


@infer_global(math.hypot)
class Math_hypot(ConcreteTemplate):
    cases = [
        signature(types.np_float64, types.np_int64, types.np_int64),
        signature(types.np_float64, types.np_uint64, types.np_uint64),
        signature(types.np_float32, types.np_float32, types.np_float32),
        signature(types.np_float64, types.np_float64, types.np_float64),
    ]


@infer_global(math.nextafter)
class Math_nextafter(ConcreteTemplate):
    cases = [
        signature(types.np_float64, types.np_float64, types.np_float64),
        signature(types.np_float32, types.np_float32, types.np_float32),
    ]


@infer_global(math.isinf)
@infer_global(math.isnan)
class Math_predicate(ConcreteTemplate):
    cases = [
        signature(types.np_bool_, types.np_int64),
        signature(types.np_bool_, types.np_uint64),
        signature(types.np_bool_, types.np_float32),
        signature(types.np_bool_, types.np_float64),
    ]


@infer_global(math.isfinite)
class Math_isfinite(Math_predicate):
    pass


@infer_global(math.pow)
class Math_pow(ConcreteTemplate):
    cases = [
        signature(types.np_float64, types.np_float64, types.np_int64),
        signature(types.np_float64, types.np_float64, types.np_uint64),
        signature(types.np_float32, types.np_float32, types.np_float32),
        signature(types.np_float64, types.np_float64, types.np_float64),
    ]


@infer_global(math.gcd)
class Math_gcd(ConcreteTemplate):
    cases = [
        signature(types.np_int64, types.np_int64, types.np_int64),
        signature(types.np_int32, types.np_int32, types.np_int32),
        signature(types.np_int16, types.np_int16, types.np_int16),
        signature(types.np_int8, types.np_int8, types.np_int8),
        signature(types.np_uint64, types.np_uint64, types.np_uint64),
        signature(types.np_uint32, types.np_uint32, types.np_uint32),
        signature(types.np_uint16, types.np_uint16, types.np_uint16),
        signature(types.np_uint8, types.np_uint8, types.np_uint8),
    ]


@infer_global(math.frexp)
class Math_frexp(ConcreteTemplate):
    cases = [
        signature(types.Tuple((types.np_float64, types.np_intc)), types.np_float64),
        signature(types.Tuple((types.np_float32, types.np_intc)), types.np_float32),
    ]

@infer_global(math.ldexp)
class Math_ldexp(ConcreteTemplate):
    cases = [
        signature(types.np_float64, types.np_float64, types.np_intc),
        signature(types.np_float32, types.np_float32, types.np_intc),
    ]
