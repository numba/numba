from __future__ import print_function, absolute_import, division

from numba import types, cgutils

from .imputils import lower_cast


def always_return_true_impl(context, builder, sig, args):
    return cgutils.true_bit


def always_return_false_impl(context, builder, sig, args):
    return cgutils.false_bit


@lower_cast(types.Optional, types.Optional)
def optional_to_optional(context, builder, fromty, toty, val):
    """
    The handling of optional->optional cast must be special cased for
    correct propagation of None value.  Given type T and U. casting of
    T? to U? (? denotes optional) should always succeed.   If the from-value
    is None, the None value the casted value (U?) should be None; otherwise,
    the from-value is casted to U. This is different from casting T? to U,
    which requires the from-value must not be None.
    """
    optval = context.make_helper(builder, fromty, value=val)
    validbit = cgutils.as_bool_bit(builder, optval.valid)
    # Create uninitialized optional value
    outoptval = context.make_helper(builder, toty)

    with builder.if_else(validbit) as (is_valid, is_not_valid):
        with is_valid:
            # Cast internal value
            outoptval.valid = cgutils.true_bit
            outoptval.data = context.cast(builder, optval.data,
                                          fromty.type, toty.type)

        with is_not_valid:
            # Store None to result
            outoptval.valid = cgutils.false_bit
            outoptval.data = cgutils.get_null_value(
                outoptval.data.type)

    return outoptval._getvalue()


@lower_cast(types.Any, types.Optional)
def any_to_optional(context, builder, fromty, toty, val):
    if fromty == types.none:
        return context.make_optional_none(builder, toty.type)
    else:
        val = context.cast(builder, val, fromty, toty.type)
        return context.make_optional_value(builder, toty.type, val)


@lower_cast(types.Optional, types.Any)
def optional_to_any(context, builder, fromty, toty, val):
    optval = context.make_helper(builder, fromty, value=val)
    validbit = cgutils.as_bool_bit(builder, optval.valid)
    with builder.if_then(builder.not_(validbit), likely=False):
        msg = "expected %s, got None" % (fromty.type,)
        context.call_conv.return_user_exc(builder, TypeError, (msg,))

    return optval.data
