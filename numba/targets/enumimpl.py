"""
Implementation of enums.
"""

from llvmlite import ir

from .imputils import (lower_builtin, lower_getattr, lower_getattr_generic,
                       iternext_impl, impl_ret_borrowed, impl_ret_untracked)
from .. import typing, types, cgutils


@lower_builtin('==', types.EnumMember, types.EnumMember)
@lower_builtin('is', types.EnumMember, types.EnumMember)
def enum_eq(context, builder, sig, args):
    tu, tv = sig.args
    u, v = args
    res = context.generic_compare(builder, "==",
                                  (tu.dtype, tv.dtype), (u, v))
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin('!=', types.EnumMember, types.EnumMember)
def enum_eq(context, builder, sig, args):
    tu, tv = sig.args
    u, v = args
    res = context.generic_compare(builder, "!=",
                                  (tu.dtype, tv.dtype), (u, v))
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_getattr(types.EnumMember, 'value')
def enum_value(context, builder, ty, val):
    return val

@lower_getattr_generic(types.EnumClass)
def enum_class_lookup(context, builder, ty, val, attr):
    """
    Return an enum member by name.
    """
    member = getattr(ty.instance_class, attr)
    return context.get_constant_generic(builder, ty.dtype, member.value)
