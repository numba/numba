"""
Implementation of the range object for fixed-size integers.
"""

import llvmlite.llvmpy.core as lc

from numba import types, cgutils
from numba.targets.imputils import (lower_builtin, lower_cast,
                                    iterator_impl, impl_ret_untracked)


def make_range_iterator(typ):
    """
    Return the Structure representation of the given *typ* (an
    instance of types.RangeIteratorType).
    """
    return cgutils.create_struct_proxy(typ)


def make_range_impl(range_state_type, range_iter_type, int_type):
    RangeState = cgutils.create_struct_proxy(range_state_type)

    @lower_builtin(range, int_type)
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

            startptr = cgutils.alloca_once(builder, start.type)
            builder.store(start, startptr)

            countptr = cgutils.alloca_once(builder, start.type)

            self.iter = startptr
            self.stop = stop
            self.step = step
            self.count = countptr

            diff = builder.sub(stop, start)
            zero = context.get_constant(int_type, 0)
            one = context.get_constant(int_type, 1)
            pos_diff = builder.icmp(lc.ICMP_SGT, diff, zero)
            pos_step = builder.icmp(lc.ICMP_SGT, step, zero)
            sign_differs = builder.xor(pos_diff, pos_step)
            zero_step = builder.icmp(lc.ICMP_EQ, step, zero)

            with cgutils.if_unlikely(builder, zero_step):
                # step shouldn't be zero
                context.call_conv.return_user_exc(builder, ValueError,
                                                  ("range() arg 3 must not be zero",))

            with builder.if_else(sign_differs) as (then, orelse):
                with then:
                    builder.store(zero, self.count)

                with orelse:
                    rem = builder.srem(diff, step)
                    rem = builder.select(pos_diff, rem, builder.neg(rem))
                    uneven = builder.icmp(lc.ICMP_SGT, rem, zero)
                    newcount = builder.add(builder.sdiv(diff, step),
                                           builder.select(uneven, one, zero))
                    builder.store(newcount, self.count)

            return self

        def iternext(self, context, builder, result):
            zero = context.get_constant(int_type, 0)
            countptr = self.count
            count = builder.load(countptr)
            is_valid = builder.icmp(lc.ICMP_SGT, count, zero)
            result.set_valid(is_valid)

            with builder.if_then(is_valid):
                value = builder.load(self.iter)
                result.yield_(value)
                one = context.get_constant(int_type, 1)

                builder.store(builder.sub(count, one, flags=["nsw"]), countptr)
                builder.store(builder.add(value, self.step), self.iter)


make_range_impl(types.range_state32_type, types.range_iter32_type, types.int32)
make_range_impl(types.range_state64_type, types.range_iter64_type, types.int64)
make_range_impl(types.unsigned_range_state64_type, types.unsigned_range_iter64_type,
                types.uint64)


@lower_cast(types.RangeType, types.RangeType)
def range_to_range(context, builder, fromty, toty, val):
    olditems = cgutils.unpack_tuple(builder, val, 3)
    items = [context.cast(builder, v, fromty.dtype, toty.dtype)
             for v in olditems]
    return cgutils.make_anonymous_struct(builder, items)
