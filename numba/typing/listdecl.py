from __future__ import absolute_import, print_function

import random

import numpy as np

from .. import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
                        Registry, signature, bound_function)
from .builtins import normalize_index


registry = Registry()
builtin = registry.register
builtin_global = registry.register_global
builtin_attr = registry.register_attr


class ListBuiltin(AbstractTemplate):
    key = list

    def generic(self, args, kws):
        assert not kws
        if args:
            iterable, = args
            if isinstance(iterable, types.IterableType):
                dtype = iterable.iterator_type.yield_type
                return signature(types.List(dtype), iterable)

builtin_global(list, types.Function(ListBuiltin))


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

    @bound_function("list.append")
    def resolve_append(self, list, args, kws):
        item, = args
        assert not kws
        unified = self.context.unify_pairs(list.dtype, item)
        sig = signature(types.none, unified)
        sig.recvr = types.List(unified)
        return sig


# XXX Should there be a base Sequence type for plain 1d sequences?

@builtin
class GetItemList(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        list, idx = args
        if isinstance(list, types.List):
            return signature(list.dtype, list, normalize_index(idx))


@builtin
class GetItemList(AbstractTemplate):
    key = "setitem"

    def generic(self, args, kws):
        list, idx, value = args
        if isinstance(list, types.List):
            return signature(types.none, list, normalize_index(idx), list.dtype)
