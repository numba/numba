import cmath

from numba import types, utils
from numba.typing.templates import (AbstractTemplate, ConcreteTemplate,
                                    signature, Registry, bound_function)

registry = Registry()
builtin_global = registry.register_global

# TODO: support non-complex arguments (floats and ints)

@builtin_global(cmath.acos)
@builtin_global(cmath.acosh)
@builtin_global(cmath.asin)
@builtin_global(cmath.asinh)
@builtin_global(cmath.atan)
@builtin_global(cmath.atanh)
@builtin_global(cmath.cos)
@builtin_global(cmath.cosh)
@builtin_global(cmath.exp)
@builtin_global(cmath.log10)
@builtin_global(cmath.sin)
@builtin_global(cmath.sinh)
@builtin_global(cmath.sqrt)
@builtin_global(cmath.tan)
@builtin_global(cmath.tanh)
class CMath_unary(ConcreteTemplate):
    cases = [signature(tp, tp) for tp in types.complex_domain]


@builtin_global(cmath.isinf)
@builtin_global(cmath.isnan)
class CMath_predicate(ConcreteTemplate):
    cases = [signature(types.boolean, tp) for tp in types.complex_domain]


if utils.PYVERSION >= (3, 2):
    @builtin_global(cmath.isfinite)
    class CMath_isfinite(CMath_predicate):
        pass


@builtin_global(cmath.log)
class Cmath_log(ConcreteTemplate):
    # unary cmath.log()
    cases = [signature(tp, tp) for tp in types.complex_domain]
    # binary cmath.log()
    cases += [signature(tp, tp, tp) for tp in types.complex_domain]


@builtin_global(cmath.phase)
class Cmath_phase(ConcreteTemplate):
    cases = [signature(tp, types.complex128) for tp in [types.float64]]
    cases += [signature(types.float32, types.complex64)]


@builtin_global(cmath.polar)
class Cmath_polar(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [tp] = args
        if tp in types.complex_domain:
            float_type = tp.underlying_float
            return signature(types.UniTuple(float_type, 2), tp)


@builtin_global(cmath.rect)
class Cmath_rect(ConcreteTemplate):
    cases = [signature(types.complex128, tp, tp)
             for tp in [types.float64]]
    cases += [signature(types.complex64, types.float32, types.float32)]
