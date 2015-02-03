from __future__ import absolute_import, print_function

import random

from .. import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
                        Registry, signature)


registry = Registry()
builtin = registry.register
builtin_global = registry.register_global
builtin_attr = registry.register_attr


# random.random(), random.seed() etc. are not plain functions, they are bound
# methods of a private object.  We have to be careful to use a well-known
# object (e.g. the string "random.seed") as a key, not the bound method itself.

_int_types = [types.int32, types.int64]
# Should we support float32?
_float_types = [types.float64]

@registry.resolves_global(random.getrandbits, typing_key="random.getrandbits")
class Random_getrandbits(ConcreteTemplate):
    cases = [signature(types.uint64, types.uint8)]

@registry.resolves_global(random.gauss, typing_key="random.gauss")
class Random_gauss(ConcreteTemplate):
    cases = [signature(tp, tp, tp) for tp in _float_types]

@registry.resolves_global(random.random, typing_key="random.random")
class Random_random(ConcreteTemplate):
    cases = [signature(types.float64)]

@registry.resolves_global(random.randint, typing_key="random.randint")
class Random_randint(ConcreteTemplate):
    _types = [types.int32, types.int64]
    cases = [signature(tp, tp, tp) for tp in _int_types]

@registry.resolves_global(random.randrange, typing_key="random.randrange")
class Random_randrange(ConcreteTemplate):
    cases = [signature(tp, tp) for tp in _int_types]
    cases += [signature(tp, tp, tp) for tp in _int_types]
    cases += [signature(tp, tp, tp, tp) for tp in _int_types]

@registry.resolves_global(random.seed, typing_key="random.seed")
class Random_seed(ConcreteTemplate):
    cases = [signature(types.void, types.uint32)]

@registry.resolves_global(random.uniform, typing_key="random.uniform")
class Random_uniform(ConcreteTemplate):
    cases = [signature(tp, tp, tp) for tp in _float_types]
