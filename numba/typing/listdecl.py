from __future__ import absolute_import, print_function

import random

import numpy as np

from .. import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
                        Registry, signature)


registry = Registry()
builtin = registry.register
builtin_global = registry.register_global
builtin_attr = registry.register_attr


@builtin
class Len(AbstractTemplate):
    key = types.len_type

    def generic(self, args, kws):
        assert not kws
        (val,) = args
        if isinstance(val, (types.List)):
            return signature(types.intp, val)
