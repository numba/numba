from __future__ import absolute_import, print_function

import random

import numpy as np

from .. import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
                        Registry, signature, bound_function)


registry = Registry()
builtin = registry.register
builtin_global = registry.register_global
builtin_attr = registry.register_attr


@builtin
class ListLen(AbstractTemplate):
    key = types.len_type

    def generic(self, args, kws):
        assert not kws
        (val,) = args
        if isinstance(val, (types.List)):
            return signature(types.intp, val)


@builtin_attr
class ListAttribute(AttributeTemplate):
    key = types.List

    @bound_function("list.pop")
    def resolve_pop(self, list, args, kws):
        assert not args
        assert not kws
        return signature(list.dtype)
