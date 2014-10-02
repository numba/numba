import cmath

from numba import types, utils
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    signature, Registry, bound_function)

registry = Registry()
builtin_attr = registry.register_attr
builtin_global = registry.register_global


# TODO: support non-complex arguments (floats and ints)

class CMath_unary(ConcreteTemplate):
    cases = [signature(tp, tp) for tp in types.complex_domain]


class CMath_predicate(ConcreteTemplate):
    cases = [signature(types.boolean, tp) for tp in types.complex_domain]


class CMath_isnan(CMath_predicate):
    key = cmath.isnan

class CMath_isinf(CMath_predicate):
    key = cmath.isinf

if utils.PYVERSION > (3, 2):
    class CMath_isfinite(CMath_predicate):
        key = cmath.isfinite


builtin_global(cmath, types.Module(cmath))
builtin_global(cmath.isnan, types.Function(CMath_isnan))
builtin_global(cmath.isinf, types.Function(CMath_isinf))
if utils.PYVERSION > (3, 2):
    builtin_global(cmath.isfinite, types.Function(CMath_isfinite))
