"""
Implement slices and various slice computations.
"""

import itertools

from llvmlite import ir

from numba.six.moves import zip_longest
from numba import cgutils, types, typing
from .imputils import (lower_builtin, lower_getattr,
                       iternext_impl, impl_ret_borrowed,
                       impl_ret_new_ref, impl_ret_untracked)


def fix_index(builder, idx, size):
    """
    Fix negative index by adding *size* to it.  Positive
    indices are left untouched.
    """
    is_negative = builder.icmp_signed('<', idx, ir.Constant(size.type, 0))
    wrapped_index = builder.add(idx, size)
    return builder.select(is_negative, wrapped_index, idx)


def fix_slice(builder, slice, size):
    """
    Fix *slice* start and stop to be valid (inclusive and exclusive, resp)
    indexing bounds for a sequence of the given *size*.
    """
    # See PySlice_GetIndicesEx()
    zero = ir.Constant(size.type, 0)
    minus_one = ir.Constant(size.type, -1)

    def fix_bound(bound_name, lower_repl, upper_repl):
        bound = getattr(slice, bound_name)
        bound = fix_index(builder, bound, size)
        # Store value
        setattr(slice, bound_name, bound)
        # Still negative? => clamp to lower_repl
        underflow = builder.icmp_signed('<', bound, zero)
        with builder.if_then(underflow, likely=False):
            setattr(slice, bound_name, lower_repl)
        # Greater than size? => clamp to upper_repl
        overflow = builder.icmp_signed('>=', bound, size)
        with builder.if_then(overflow, likely=False):
            setattr(slice, bound_name, upper_repl)

    with builder.if_else(cgutils.is_neg_int(builder, slice.step)) as (if_neg_step, if_pos_step):
        with if_pos_step:
            # < 0 => 0; >= size => size
            fix_bound('start', zero, size)
            fix_bound('stop', zero, size)
        with if_neg_step:
            # < 0 => -1; >= size => size - 1
            lower = minus_one
            upper = builder.add(size, minus_one)
            fix_bound('start', lower, upper)
            fix_bound('stop', lower, upper)


def get_slice_length(builder, slicestruct):
    """
    Given a slice, compute the number of indices it spans, i.e. the
    number of iterations that for_range_slice() will execute.

    Pseudo-code:
        assert step != 0
        if step > 0:
            if stop <= start:
                return 0
            else:
                return (stop - start - 1) // step + 1
        else:
            if stop >= start:
                return 0
            else:
                return (stop - start + 1) // step + 1

    (see PySlice_GetIndicesEx() in CPython)
    """
    start = slicestruct.start
    stop = slicestruct.stop
    step = slicestruct.step
    one = ir.Constant(start.type, 1)
    zero = ir.Constant(start.type, 0)

    is_step_negative = cgutils.is_neg_int(builder, step)
    delta = builder.sub(stop, start)

    # Nominal case
    pos_dividend = builder.sub(delta, one)
    neg_dividend = builder.add(delta, one)
    dividend  = builder.select(is_step_negative, neg_dividend, pos_dividend)
    nominal_length = builder.add(one, builder.sdiv(dividend, step))

    # Catch zero length
    is_zero_length = builder.select(is_step_negative,
                                    builder.icmp_signed('>=', delta, zero),
                                    builder.icmp_signed('<=', delta, zero))

    # Clamp to 0 if is_zero_length
    return builder.select(is_zero_length, zero, nominal_length)


def fix_stride(builder, slice, stride):
    """
    Fix the given stride for the slice's step.
    """
    return builder.mul(slice.step, stride)

def guard_invalid_slice(context, builder, typ, slicestruct):
    """
    Guard against *slicestruct* having a zero step (and raise ValueError).
    """
    if typ.has_step:
        cgutils.guard_null(context, builder, slicestruct.step,
                           (ValueError, "slice step cannot be zero"))


def get_defaults(context):
    """
    Get the default values for a slice's members:
    (start for positive step, start for negative step,
     stop for positive step, stop for negative step, step)
    """
    maxint = (1 << (context.address_size - 1)) - 1
    return (0, maxint, maxint, - maxint - 1, 1)


#---------------------------------------------------------------------------
# The slice structure

@lower_builtin(slice, types.VarArg(types.Any))
def slice_constructor_impl(context, builder, sig, args):
    default_start_pos, default_start_neg, default_stop_pos, default_stop_neg, default_step = \
        [context.get_constant(types.intp, x) for x in get_defaults(context)]

    # Fetch non-None arguments
    slice_args = [None] * 3
    for i, (ty, val) in enumerate(zip(sig.args, args)):
        if ty is types.none:
            slice_args[i] = None
        else:
            slice_args[i] = val

    # Fill omitted arguments
    def get_arg_value(i, default):
        val = slice_args[i]
        if val is None:
            return default
        else:
            return val

    step = get_arg_value(2, default_step)
    is_step_negative = builder.icmp_signed('<', step,
                                           context.get_constant(types.intp, 0))
    default_stop = builder.select(is_step_negative,
                                  default_stop_neg, default_stop_pos)
    default_start = builder.select(is_step_negative,
                                   default_start_neg, default_start_pos)
    stop = get_arg_value(1, default_stop)
    start = get_arg_value(0, default_start)

    ty = sig.return_type
    sli = context.make_helper(builder, sig.return_type)
    sli.start = start
    sli.stop = stop
    sli.step = step

    res = sli._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_getattr(types.SliceType, "start")
def slice_start_impl(context, builder, typ, value):
    sli = context.make_helper(builder, typ, value)
    return sli.start

@lower_getattr(types.SliceType, "stop")
def slice_stop_impl(context, builder, typ, value):
    sli = context.make_helper(builder, typ, value)
    return sli.stop

@lower_getattr(types.SliceType, "step")
def slice_step_impl(context, builder, typ, value):
    if typ.has_step:
        sli = context.make_helper(builder, typ, value)
        return sli.step
    else:
        return context.get_constant(types.intp, 1)
