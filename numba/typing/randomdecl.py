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

@registry.resolves_global(random.getrandbits, typing_key="random.getrandbits")
class Random_getrandbits(ConcreteTemplate):
    cases = [signature(types.uint64, types.uint32)]

@registry.resolves_global(random.gauss, typing_key="random.gauss")
class Random_gauss(ConcreteTemplate):
    # Should we have another case for float32?
    cases = [signature(types.float64, types.float64, types.float64)]

@registry.resolves_global(random.random, typing_key="random.random")
class Random_random(ConcreteTemplate):
    cases = [signature(types.float64)]

@registry.resolves_global(random.seed, typing_key="random.seed")
class Random_seed(ConcreteTemplate):
    cases = [signature(types.void, types.uint32)]
