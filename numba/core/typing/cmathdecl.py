import cmath

from numba.core import types, utils
from numba.core.typing.templates import (AbstractTemplate, ConcreteTemplate,
                                    signature, Registry)

registry = Registry()
infer_global = registry.register_global

# TODO: support non-complex arguments (floats and ints)


@infer_global(cmath.acosh)
@infer_global(cmath.cosh)
@infer_global(cmath.exp)
@infer_global(cmath.log10)
@infer_global(cmath.sinh)
@infer_global(cmath.tanh)
class CMath_unary(ConcreteTemplate):
    cases = [signature(tp, tp) for tp in sorted(types.complex_domain)]


@infer_global(cmath.phase)
class Cmath_phase(ConcreteTemplate):
    cases = [signature(tp, types.complex128) for tp in [types.float64]]
    cases += [signature(types.float32, types.complex64)]


@infer_global(cmath.polar)
class Cmath_polar(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [tp] = args
        if tp in types.complex_domain:
            float_type = tp.underlying_float
            return signature(types.UniTuple(float_type, 2), tp)


@infer_global(cmath.rect)
class Cmath_rect(ConcreteTemplate):
    cases = [signature(types.complex128, tp, tp)
             for tp in [types.float64]]
    cases += [signature(types.complex64, types.float32, types.float32)]
