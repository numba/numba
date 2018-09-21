from __future__ import absolute_import, print_function

from .. import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
                        CallableTemplate,  Registry, signature, bound_function,
                        make_callable_template)
# Ensure list is typed as a collection as well
from . import collections


registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr


@infer_global(list)
class ListBuiltin(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if args:
            iterable, = args
            if isinstance(iterable, types.IterableType):
                dtype = iterable.iterator_type.yield_type
                return signature(types.List(dtype), iterable)
        else:
            return signature(types.List(types.undefined))


@infer_global(sorted)
class SortedBuiltin(CallableTemplate):

    def generic(self):
        def typer(iterable, reverse=None):
            if not isinstance(iterable, types.IterableType):
                return
            if (reverse is not None and
                not isinstance(reverse, types.Boolean)):
                return
            return types.List(iterable.iterator_type.yield_type)

        return typer


@infer_getattr
class ListAttribute(AttributeTemplate):
    key = types.List

    # NOTE: some of these should be Sequence / MutableSequence methods

    @bound_function("list.append")
    def resolve_append(self, list, args, kws):
        item, = args
        assert not kws
        unified = self.context.unify_pairs(list.dtype, item)
        if unified is not None:
            sig = signature(types.none, unified)
            sig.recvr = list.copy(dtype=unified)
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
        if unified is not None:
            sig = signature(types.none, iterable)
            sig.recvr = list.copy(dtype=unified)
            return sig

    @bound_function("list.index")
    def resolve_index(self, list, args, kws):
        assert not kws
        if len(args) == 1:
            return signature(types.intp, list.dtype)
        elif len(args) == 2:
            if isinstance(args[1], types.Integer):
                return signature(types.intp, list.dtype, types.intp)
        elif len(args) == 3:
            if (isinstance(args[1], types.Integer)
                and isinstance(args[2], types.Integer)):
                return signature(types.intp, list.dtype, types.intp, types.intp)

    @bound_function("list.insert")
    def resolve_insert(self, list, args, kws):
        idx, item = args
        assert not kws
        if isinstance(idx, types.Integer):
            unified = self.context.unify_pairs(list.dtype, item)
            if unified is not None:
                sig = signature(types.none, types.intp, unified)
                sig.recvr = list.copy(dtype=unified)
                return sig

    @bound_function("list.pop")
    def resolve_pop(self, list, args, kws):
        assert not kws
        if not args:
            return signature(list.dtype)
        else:
            idx, = args
            if isinstance(idx, types.Integer):
                return signature(list.dtype, types.intp)

    @bound_function("list.remove")
    def resolve_remove(self, list, args, kws):
        assert not kws
        if len(args) == 1:
            return signature(types.none, list.dtype)

    @bound_function("list.reverse")
    def resolve_reverse(self, list, args, kws):
        assert not args
        assert not kws
        return signature(types.none)

    def resolve_sort(self, list):
        def typer(reverse=None):
            if (reverse is not None and
                not isinstance(reverse, types.Boolean)):
                return
            return types.none

        return types.BoundFunction(make_callable_template(key="list.sort",
                                                          typer=typer,
                                                          recvr=list),
                                   list)


@infer
class AddList(AbstractTemplate):
    key = "+"

    def generic(self, args, kws):
        if len(args) == 2:
            a, b = args
            if isinstance(a, types.List) and isinstance(b, types.List):
                unified = self.context.unify_pairs(a, b)
                if unified is not None:
                    return signature(unified, a, b)


@infer
class InplaceAddList(AbstractTemplate):
    key = "+="

    def generic(self, args, kws):
        if len(args) == 2:
            a, b = args
            if isinstance(a, types.List) and isinstance(b, types.List):
                if self.context.can_convert(b.dtype, a.dtype):
                    return signature(a, a, b)


@infer
class MulList(AbstractTemplate):
    key = "*"

    def generic(self, args, kws):
        a, b = args
        if isinstance(a, types.List) and isinstance(b, types.Integer):
            return signature(a, a, types.intp)


@infer
class InplaceMulList(MulList):
    key = "*="


class ListCompare(AbstractTemplate):

    def generic(self, args, kws):
        [lhs, rhs] = args
        if isinstance(lhs, types.List) and isinstance(rhs, types.List):
            # Check element-wise comparability
            res = self.context.resolve_function_type(self.key,
                                                     (lhs.dtype, rhs.dtype), {})
            if res is not None:
                return signature(types.boolean, lhs, rhs)

@infer
class ListEq(ListCompare):
    key = '=='

@infer
class ListNe(ListCompare):
    key = '!='

@infer
class ListLt(ListCompare):
    key = '<'

@infer
class ListLe(ListCompare):
    key = '<='

@infer
class ListGt(ListCompare):
    key = '>'

@infer
class ListGe(ListCompare):
    key = '>='
