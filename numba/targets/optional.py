from __future__ import print_function, absolute_import, division
from numba import types, cgutils


def make_optional(valtype):
    """
    Return the Structure representation of a optional value
    """
    return cgutils.create_struct_proxy(types.Optional(valtype))


def always_return_true_impl(context, builder, sig, args):
    return cgutils.true_bit


def always_return_false_impl(context, builder, sig, args):
    return cgutils.false_bit
