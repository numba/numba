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


@builtin_attr
class ListAttribute(AttributeTemplate):
    key = types.List

    @bound_function("list.append")
    def resolve_append(self, list, args, kws):
        item, = args
        assert not kws
        unified = self.context.unify_pairs(list.dtype, item)
        sig = signature(types.none, unified)
        sig.recvr = types.List(unified)
        return sig

    @bound_function("list.clear")
    def resolve_clear(self, list, args, kws):
        assert not args
        assert not kws
        return signature(types.none)

    @bound_function("list.copy")
    def resolve_copy(self, list, args, kws):
        assert not args
        assert not kws
        return signature(list)

    @bound_function("list.count")
    def resolve_count(self, list, args, kws):
        item, = args
        assert not kws
        return signature(types.intp, list.dtype)

    @bound_function("list.extend")
    def resolve_extend(self, list, args, kws):
        iterable, = args
        assert not kws
        if not isinstance(iterable, types.IterableType):
            return

        dtype = iterable.iterator_type.yield_type
        unified = self.context.unify_pairs(list.dtype, dtype)
      
        sig = signature(types.none, iterable)
        sig.recvr = types.List(unified)
        return sig

    @bound_function("list.index")
    def resolve_index(self, list, args, kws):
        # XXX handle optional start, stop
        item, = args
        assert not kws
        return signature(types.intp, list.dtype)

    @bound_function("list.pop")
    def resolve_pop(self, list, args, kws):
        # XXX handle optional index
        assert not args
        assert not kws
        return signature(list.dtype)

    @bound_function("list.reverse")
    def resolve_reverse(self, list, args, kws):
        assert not args
        assert not kws
        return signature(types.none)


# XXX Should there be a base Sequence type for plain 1d sequences?

@builtin
class ListLen(AbstractTemplate):
    key = types.len_type

    def generic(self, args, kws):
        assert not kws
        (val,) = args
        if isinstance(val, (types.List)):
            return signature(types.intp, val)

@builtin
class GetItemList(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        list, idx = args
        if isinstance(list, types.List):
            idx = normalize_index(idx)
            if idx == types.slice3_type:
                return signature(list, list, idx)
            elif isinstance(idx, types.Integer):
                return signature(list.dtype, list, idx)


@builtin
class SetItemList(AbstractTemplate):
    key = "setitem"

    def generic(self, args, kws):
        list, idx, value = args
        if isinstance(list, types.List):
            return signature(types.none, list, normalize_index(idx), list.dtype)


@builtin
class InList(AbstractTemplate):
    key = "in"

    def generic(self, args, kws):
        item, list = args
        if isinstance(list, types.List):
            return signature(types.boolean, list.dtype, list)
