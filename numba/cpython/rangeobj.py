"""
Implementation of the range object for fixed-size integers.
"""

import operator

from numba import prange
from numba.core import types, cgutils, errors
from numba.cpython.listobj import ListIterInstance
from numba.np.arrayobj import make_array
from numba.core.imputils import (lower_builtin, lower_cast,
                                    iterator_impl, impl_ret_untracked)
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, overload_attribute, register_jitable
from numba.parfors.parfor import internal_prange

from llvmlite import ir as llvmir


def make_range_iterator(typ):
    """
    Return the Structure representation of the given *typ* (an
    instance of types.RangeIteratorType).
    """
    return cgutils.create_struct_proxy(typ)


def make_range_impl(int_type, range_state_type, range_iter_type):
    RangeState = cgutils.create_struct_proxy(range_state_type)

    @lower_builtin(range, int_type)
    @lower_builtin(prange, int_type)
    @lower_builtin(internal_prange, int_type)
    def range1_impl(context, builder, sig, args):
        """
        range(stop: int) -> range object
        """
        [stop] = args
        state = RangeState(context, builder)
        state.start = context.get_constant(int_type, 0)
        state.stop = stop
        state.step = context.get_constant(int_type, 1)
        return impl_ret_untracked(context,
                                  builder,
                                  range_state_type,
                                  state._getvalue())

    @lower_builtin(range, int_type, int_type)
    @lower_builtin(prange, int_type, int_type)
    @lower_builtin(internal_prange, int_type, int_type)
    def range2_impl(context, builder, sig, args):
        """
        range(start: int, stop: int) -> range object
        """
        start, stop = args
        state = RangeState(context, builder)
        state.start = start
        state.stop = stop
        state.step = context.get_constant(int_type, 1)
        return impl_ret_untracked(context,
                                  builder,
                                  range_state_type,
                                  state._getvalue())

    @lower_builtin(range, int_type, int_type, int_type)
    @lower_builtin(prange, int_type, int_type, int_type)
    @lower_builtin(internal_prange, int_type, int_type, int_type)
    def range3_impl(context, builder, sig, args):
        """
        range(start: int, stop: int, step: int) -> range object
        """
        [start, stop, step] = args
        state = RangeState(context, builder)
        state.start = start
        state.stop = stop
        state.step = step
        return impl_ret_untracked(context,
                                  builder,
                                  range_state_type,
                                  state._getvalue())

    @lower_builtin(len, range_state_type)
    def range_len(context, builder, sig, args):
        """
        len(range)
        """
        (value,) = args
        state = RangeState(context, builder, value)
        res = RangeIter.from_range_state(context, builder, state)
        return impl_ret_untracked(context, builder, int_type, builder.load(res.count))

    @lower_builtin('getiter', range_state_type)
    def getiter_range32_impl(context, builder, sig, args):
        """
        range.__iter__
        """
        (value,) = args
        state = RangeState(context, builder, value)
        res = RangeIter.from_range_state(context, builder, state)._getvalue()
        return impl_ret_untracked(context, builder, range_iter_type, res)

    @iterator_impl(range_state_type, range_iter_type)
    class RangeIter(make_range_iterator(range_iter_type)):

        @classmethod
        def from_range_state(cls, context, builder, state):
            """
            Create a RangeIter initialized from the given RangeState *state*.
            """
            self = cls(context, builder)
            start = state.start
            stop = state.stop
            step = state.step

            lltype = stop.type
            # guard zero step
            zero = context.get_constant(int_type, 0)
            zero_step = builder.icmp_unsigned('==', step, zero)

            with cgutils.if_unlikely(builder, zero_step):
                # step shouldn't be zero
                context.call_conv.return_user_exc(builder, ValueError,
                                                  ("range() arg 3 must not be zero",))
            # diff = stop - start
            diff = builder.sub(stop, start, name='diff')
            # endadj = -1 if step < 0 else +1
            negative = builder.icmp_signed('<', step, lltype(0))
            endadj = builder.select(negative, lltype(-1), lltype(+1),
                                    name='endadj')
            # adjust start
            adj_start = lltype(0)
            # adj_stop = (diff + (step - endadj)) // step
            adj_stop = builder.sdiv(
                builder.add(diff, builder.sub(step, endadj)),
                step,
                name="adj_stop",
            )

            # store range states
            startptr = cgutils.alloca_once(builder, start.type)
            builder.store(adj_start, startptr)

            self.iter = startptr
            self.start = start
            self.adj_stop = adj_stop
            self.scale = step

            # handle count
            # It's done in a way that help LLVM optimize-away all these.
            # It is rare for users to query the len() of a range.
            countptr = cgutils.alloca_once(builder, start.type)
            self.count = countptr

            one = context.get_constant(int_type, 1)
            pos_diff = builder.icmp_signed('>', diff, zero)
            pos_step = builder.icmp_signed('>', step, zero)
            sign_differs = builder.xor(pos_diff, pos_step)
            with builder.if_else(sign_differs) as (then, orelse):
                with then:
                    builder.store(zero, self.count)

                with orelse:
                    rem = builder.srem(diff, step)
                    rem = builder.select(pos_diff, rem, builder.neg(rem))
                    uneven = builder.icmp_signed('>', rem, zero)
                    newcount = builder.add(builder.sdiv(diff, step),
                                           builder.select(uneven, one, zero))
                    builder.store(newcount, self.count)

            return self

        def iternext(self, context, builder, result):
            indvar = builder.load(self.iter)

            # Compute the next indvar.
            # Handle overflow by extending to a bigger int type.
            # Sadly, loop-vectorizer does not handle llvm.sadd.with.overflow.
            ext_type = llvmir.IntType(indvar.type.width + 1)

            ext_indvar = builder.sext(indvar, ext_type)
            ext_next_indvar = builder.add(ext_indvar, ext_indvar.type(1))

            of = builder.trunc(
                builder.lshr(
                    ext_next_indvar,
                    ext_next_indvar.type(ext_type.width - 1),
                ),
                llvmir.IntType(1),
                name='overflow',
            )

            next_indvar = builder.trunc(ext_next_indvar, indvar.type,
                                        name='next_indvar')
            is_valid = builder.and_(
                builder.not_(of),
                builder.icmp_signed("<", indvar, self.adj_stop),
            )
            result.set_valid(is_valid)

            with builder.if_then(is_valid):
                # start + (indvar * scale)
                result.yield_(
                    builder.add(
                        self.start,
                        builder.mul(indvar, self.scale),
                        name='scaled_indvar',
                    ),
                )
                builder.store(next_indvar, self.iter)


range_impl_map = {
    types.int32 : (types.range_state32_type, types.range_iter32_type),
    types.int64 : (types.range_state64_type, types.range_iter64_type),
    types.uint64 : (types.unsigned_range_state64_type, types.unsigned_range_iter64_type)
}

for int_type, state_types in range_impl_map.items():
    make_range_impl(int_type, *state_types)

@lower_cast(types.RangeType, types.RangeType)
def range_to_range(context, builder, fromty, toty, val):
    olditems = cgutils.unpack_tuple(builder, val, 3)
    items = [context.cast(builder, v, fromty.dtype, toty.dtype)
             for v in olditems]
    return cgutils.make_anonymous_struct(builder, items)

@intrinsic
def length_of_iterator(typingctx, val):
    """
    An implementation of len(iter) for internal use.
    Primary use is for array comprehensions (see inline_closurecall).
    """
    if isinstance(val, types.RangeIteratorType):
        val_type = val.yield_type
        def codegen(context, builder, sig, args):
            (value,) = args
            iter_type = range_impl_map[val_type][1]
            iterobj = cgutils.create_struct_proxy(iter_type)(context, builder, value)
            int_type = iterobj.count.type
            return impl_ret_untracked(context, builder, int_type, builder.load(iterobj.count))
        return signature(val_type, val), codegen
    elif isinstance(val, types.ListIter):
        def codegen(context, builder, sig, args):
            (value,) = args
            intp_t = context.get_value_type(types.intp)
            iterobj = ListIterInstance(context, builder, sig.args[0], value)
            return impl_ret_untracked(context, builder, intp_t, iterobj.size)
        return signature(types.intp, val), codegen
    elif isinstance(val, types.ArrayIterator):
        def  codegen(context, builder, sig, args):
            (iterty,) = sig.args
            (value,) = args
            intp_t = context.get_value_type(types.intp)
            iterobj = context.make_helper(builder, iterty, value=value)
            arrayty = iterty.array_type
            ary = make_array(arrayty)(context, builder, value=iterobj.array)
            shape = cgutils.unpack_tuple(builder, ary.shape)
            # array iterates along the outer dimension
            return impl_ret_untracked(context, builder, intp_t, shape[0])
        return signature(types.intp, val), codegen
    elif isinstance(val, types.UniTupleIter):
        def codegen(context, builder, sig, args):
            (iterty,) = sig.args
            tuplety = iterty.container
            intp_t = context.get_value_type(types.intp)
            count_const = intp_t(tuplety.count)
            return impl_ret_untracked(context, builder, intp_t, count_const)

        return signature(types.intp, val), codegen
    elif isinstance(val, types.ListTypeIteratorType):
        def codegen(context, builder, sig, args):
            (value,) = args
            intp_t = context.get_value_type(types.intp)
            from numba.typed.listobject import ListIterInstance
            iterobj = ListIterInstance(context, builder, sig.args[0], value)
            return impl_ret_untracked(context, builder, intp_t, iterobj.size)
        return signature(types.intp, val), codegen
    else:
        msg = ('Unsupported iterator found in array comprehension, try '
               'preallocating the array and filling manually.')
        raise errors.TypingError(msg)

def make_range_attr(index, attribute):
    @intrinsic
    def rangetype_attr_getter(typingctx, a):
        if isinstance(a, types.RangeType):
            def codegen(context, builder, sig, args):
                (val,) = args
                items = cgutils.unpack_tuple(builder, val, 3)
                return impl_ret_untracked(context, builder, sig.return_type,
                                          items[index])
            return signature(a.dtype, a), codegen

    @overload_attribute(types.RangeType, attribute)
    def range_attr(rnge):
        def get(rnge):
            return rangetype_attr_getter(rnge)
        return get


@register_jitable
def impl_contains_helper(robj, val):
    if robj.step > 0 and (val < robj.start or val >= robj.stop):
        return False
    elif robj.step < 0 and (val <= robj.stop or val > robj.start):
        return False

    return ((val - robj.start) % robj.step) == 0


@overload(operator.contains)
def impl_contains(robj, val):
    def impl_false(robj, val):
        return False

    if not isinstance(robj, types.RangeType):
        return

    elif isinstance(val, (types.Integer, types.Boolean)):
        return impl_contains_helper

    elif isinstance(val, types.Float):
        def impl(robj, val):
            if val % 1 != 0:
                return False
            else:
                return impl_contains_helper(robj, int(val))
        return impl

    elif isinstance(val, types.Complex):
        def impl(robj, val):
            if val.imag != 0:
                return False
            elif val.real % 1 != 0:
                return False
            else:
                return impl_contains_helper(robj, int(val.real))
        return impl

    elif not isinstance(val, types.Number):
        return impl_false


for ix, attr in enumerate(('start', 'stop', 'step')):
    make_range_attr(index=ix, attribute=attr)
