from __future__ import print_function, absolute_import, division

from numba import types, cgutils

from .imputils import lower_cast


def make_optional(valtype):
    """
    Return the Structure representation of a optional value
    """
    return cgutils.create_struct_proxy(types.Optional(valtype))


def always_return_true_impl(context, builder, sig, args):
    return cgutils.true_bit


def always_return_false_impl(context, builder, sig, args):
    return cgutils.false_bit


@lower_cast(types.Any, types.Optional)
def any_to_optional(context, builder, fromty, toty, val):
    if fromty == types.none:
        return context.make_optional_none(builder, toty.type)
    else:
        val = context.cast(builder, val, fromty, toty.type)
        return context.make_optional_value(builder, toty.type, val)

@lower_cast(types.Optional, types.Any)
def optional_to_any(context, builder, fromty, toty, val):
    optty = context.make_optional(fromty)
    optval = optty(context, builder, value=val)
    validbit = cgutils.as_bool_bit(builder, optval.valid)
    with builder.if_then(builder.not_(validbit), likely=False):
        msg = "expected %s, got None" % (fromty.type,)
        context.call_conv.return_user_exc(builder, TypeError, (msg,))

    return optval.data
