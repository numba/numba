import cmath

from numba import types, utils
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    signature, Registry, bound_function)

registry = Registry()

# TODO: support non-complex arguments (floats and ints)

class CMath_unary(ConcreteTemplate):
    cases = [signature(tp, tp) for tp in types.complex_domain]

@registry.resolves_global(cmath.isinf)
@registry.resolves_global(cmath.isnan)
class CMath_predicate(ConcreteTemplate):
    cases = [signature(types.boolean, tp) for tp in types.complex_domain]


if utils.PYVERSION > (3, 2):
    @registry.resolves_global(cmath.isfinite)
    class CMath_isfinite(CMath_predicate):
        pass
