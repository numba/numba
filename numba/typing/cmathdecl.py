import cmath

from numba import types, utils
from numba.typing.templates import (AbstractTemplate, ConcreteTemplate,
                                    signature, Registry, bound_function)

registry = Registry()

# TODO: support non-complex arguments (floats and ints)

@registry.resolves_global(cmath.exp)
@registry.resolves_global(cmath.log10)
class CMath_unary(ConcreteTemplate):
    cases = [signature(tp, tp) for tp in types.complex_domain]


@registry.resolves_global(cmath.isinf)
@registry.resolves_global(cmath.isnan)
class CMath_predicate(ConcreteTemplate):
    cases = [signature(types.boolean, tp) for tp in types.complex_domain]


if utils.PYVERSION >= (3, 2):
    @registry.resolves_global(cmath.isfinite)
    class CMath_isfinite(CMath_predicate):
        pass


@registry.resolves_global(cmath.log)
class Cmath_log(ConcreteTemplate):
    # unary cmath.log()
    cases = [signature(tp, tp) for tp in types.complex_domain]
    # binary cmath.log()
    cases += [signature(tp, tp, tp) for tp in types.complex_domain]


@registry.resolves_global(cmath.phase)
class Cmath_phase(ConcreteTemplate):
    cases = [signature(tp, types.complex128) for tp in [types.float64]]
    cases += [signature(types.float32, types.complex64)]


@registry.resolves_global(cmath.polar)
class Cmath_polar(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [tp] = args
        if tp in types.complex_domain:
            float_type = tp.underlying_float
            return signature(types.UniTuple(float_type, 2), tp)


@registry.resolves_global(cmath.rect)
class Cmath_rect(ConcreteTemplate):
    cases = [signature(types.complex128, tp, tp)
             for tp in [types.float64]]
    cases += [signature(types.complex64, types.float32, types.float32)]

