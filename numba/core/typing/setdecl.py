import operator

from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
                        CallableTemplate,  Registry, signature, bound_function,
                        make_callable_template)
# Ensure set is typed as a collection as well
from numba.core.typing import collections


registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr


@infer_global(set)
class SetBuiltin(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if args:
            # set(iterable)
            iterable, = args
            if isinstance(iterable, types.IterableType):
                dtype = iterable.iterator_type.yield_type
                if isinstance(dtype, types.Hashable):
                    return signature(types.Set(dtype), iterable)
        else:
            # set()
            return signature(types.Set(types.undefined))


@infer_getattr
class SetAttribute(AttributeTemplate):
    key = types.Set

    @bound_function("set.add")
    def resolve_add(self, set, args, kws):
        item, = args
        assert not kws
        unified = self.context.unify_pairs(set.dtype, item)
        if unified is not None:
            sig = signature(types.none, unified)
            sig = sig.replace(recvr=set.copy(dtype=unified))
            return sig

    @bound_function("set.clear")
    def resolve_clear(self, set, args, kws):
        assert not kws
        if not args:
            return signature(types.none)

    @bound_function("set.copy")
    def resolve_copy(self, set, args, kws):
        assert not kws
        if not args:
            return signature(set)

    @bound_function("set.discard")
    def resolve_discard(self, set, args, kws):
        item, = args
        assert not kws
        return signature(types.none, set.dtype)

    @bound_function("set.pop")
    def resolve_pop(self, set, args, kws):
        assert not kws
        if not args:
            return signature(set.dtype)

    @bound_function("set.remove")
    def resolve_remove(self, set, args, kws):
        item, = args
        assert not kws
        return signature(types.none, set.dtype)

    @bound_function("set.update")
    def resolve_update(self, set, args, kws):
        iterable, = args
        assert not kws
        if not isinstance(iterable, types.IterableType):
            return

        dtype = iterable.iterator_type.yield_type
        unified = self.context.unify_pairs(set.dtype, dtype)
        if unified is not None:
            sig = signature(types.none, iterable)
            sig = sig.replace(recvr=set.copy(dtype=unified))
            return sig

    def _resolve_xxx_update(self, set, args, kws):
        assert not kws
        iterable, = args
        # Set arguments only supported for now
        # (note we can mix non-reflected and reflected arguments)
        if isinstance(iterable, types.Set) and iterable.dtype == set.dtype:
            return signature(types.none, iterable)

    @bound_function("set.difference_update")
    def resolve_difference_update(self, set, args, kws):
        return self._resolve_xxx_update(set, args, kws)

    @bound_function("set.intersection_update")
    def resolve_intersection_update(self, set, args, kws):
        return self._resolve_xxx_update(set, args, kws)

    @bound_function("set.symmetric_difference_update")
    def resolve_symmetric_difference_update(self, set, args, kws):
        return self._resolve_xxx_update(set, args, kws)

    def _resolve_operator(self, set, args, kws):
        assert not kws
        iterable, = args
        # Set arguments only supported for now
        # (note we can mix non-reflected and reflected arguments)
        if isinstance(iterable, types.Set) and iterable.dtype == set.dtype:
            return signature(set, iterable)

    @bound_function("set.difference")
    def resolve_difference(self, set, args, kws):
        return self._resolve_operator(set, args, kws)

    @bound_function("set.intersection")
    def resolve_intersection(self, set, args, kws):
        return self._resolve_operator(set, args, kws)

    @bound_function("set.symmetric_difference")
    def resolve_symmetric_difference(self, set, args, kws):
        return self._resolve_operator(set, args, kws)

    @bound_function("set.union")
    def resolve_union(self, set, args, kws):
        return self._resolve_operator(set, args, kws)

    def _resolve_comparator(self, set, args, kws):
        assert not kws
        arg, = args
        if arg == set:
            return signature(types.boolean, arg)

    @bound_function("set.isdisjoint")
    def resolve_isdisjoint(self, set, args, kws):
        return self._resolve_comparator(set, args, kws)

    @bound_function("set.issubset")
    def resolve_issubset(self, set, args, kws):
        return self._resolve_comparator(set, args, kws)

    @bound_function("set.issuperset")
    def resolve_issuperset(self, set, args, kws):
        return self._resolve_comparator(set, args, kws)


class SetOperator(AbstractTemplate):

    def generic(self, args, kws):
        if len(args) != 2:
            return
        a, b = args
        if (isinstance(a, types.Set) and isinstance(b, types.Set)
            and a.dtype == b.dtype):
            return signature(a, *args)


class SetComparison(AbstractTemplate):

    def generic(self, args, kws):
        if len(args) != 2:
            return
        a, b = args
        if isinstance(a, types.Set) and isinstance(b, types.Set) and a == b:
            return signature(types.boolean, *args)


for op_key in (operator.add, operator.sub, operator.and_, operator.or_, operator.xor, operator.invert):
    @infer_global(op_key)
    class ConcreteSetOperator(SetOperator):
        key = op_key


for op_key in (operator.iadd, operator.isub, operator.iand, operator.ior, operator.ixor):
    @infer_global(op_key)
    class ConcreteInplaceSetOperator(SetOperator):
        key = op_key


for op_key in (operator.eq, operator.ne, operator.lt, operator.le, operator.ge, operator.gt):
    @infer_global(op_key)
    class ConcreteSetComparison(SetComparison):
        key = op_key

