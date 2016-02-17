from __future__ import absolute_import, print_function

from .. import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
                        CallableTemplate,  Registry, signature, bound_function,
                        make_callable_template)
# Ensure set is typed as a collection as well
from . import collections


registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr


@infer_global(set)
class SetBuiltin(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if args:
            iterable, = args
            if isinstance(iterable, types.IterableType):
                dtype = iterable.iterator_type.yield_type
                if isinstance(dtype, types.Hashable):
                    return signature(types.Set(dtype), iterable)

#infer_global(set, types.Function(SetBuiltin))


@infer_getattr
class SetAttribute(AttributeTemplate):
    key = types.Set

    @bound_function("set.add")
    def resolve_add(self, set, args, kws):
        item, = args
        assert not kws
        unified = self.context.unify_pairs(set.dtype, item)
        sig = signature(types.none, unified)
        sig.recvr = set.copy(dtype=unified)
        return sig

    @bound_function("set.update")
    def resolve_update(self, set, args, kws):
        iterable, = args
        assert not kws
        if not isinstance(iterable, types.IterableType):
            return

        dtype = iterable.iterator_type.yield_type
        unified = self.context.unify_pairs(set.dtype, dtype)

        sig = signature(types.none, iterable)
        sig.recvr = set.copy(dtype=unified)
        return sig
