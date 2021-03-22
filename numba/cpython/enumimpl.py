"""
Implementation of enums.
"""
import operator

from numba.core.imputils import (lower_builtin, lower_getattr,
                                 lower_getattr_generic, lower_cast,
                                 lower_constant, impl_ret_untracked,
                                 impl_ret_borrowed)
from numba.core import types, cgutils
from numba.core.extending import overload_method


def _get_member_name(builder, member):
    return builder.extract_value(member, 0)


def _get_member_value(builder, member):
    return builder.extract_value(member, 1)


@lower_builtin(operator.eq, types.EnumMember, types.EnumMember)
def enum_eq(context, builder, sig, args):
    tu, tv = sig.args
    u, v = args
    uval, vval = _get_member_value(builder, u), _get_member_value(builder, v)
    res = context.generic_compare(builder, operator.eq,
                                  (tu.dtype, tv.dtype), (uval, vval))
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin(operator.is_, types.EnumMember, types.EnumMember)
def enum_is(context, builder, sig, args):
    tu, tv = sig.args
    u, v = args
    uval, vval = _get_member_value(builder, u), _get_member_value(builder, v)
    if tu == tv:
        res = context.generic_compare(builder, operator.eq,
                                      (tu.dtype, tv.dtype), (uval, vval))
    else:
        res = context.get_constant(sig.return_type, False)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin(operator.ne, types.EnumMember, types.EnumMember)
def enum_ne(context, builder, sig, args):
    tu, tv = sig.args
    u, v = args
    uval, vval = _get_member_value(builder, u), _get_member_value(builder, v)
    res = context.generic_compare(builder, operator.ne,
                                  (tu.dtype, tv.dtype), (uval, vval))
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_getattr(types.EnumMember, 'name')
def enum_name(context, builder, ty, val):
    res = _get_member_name(builder, val)
    return impl_ret_borrowed(context, builder, ty.ntype, res)


@lower_getattr(types.EnumMember, 'value')
def enum_value(context, builder, ty, val):
    res = _get_member_value(builder, val)
    return impl_ret_borrowed(context, builder, ty.dtype, res)


@overload_method(types.EnumMember, '__hash__')
def enum_hash(val):
    return lambda val: hash(val.name)

@overload_method(types.IntEnumMember, '__hash__')
def intenum_hash(val):
    return lambda val: hash(val.value)


@lower_cast(types.IntEnumMember, types.Integer)
def int_enum_to_int(context, builder, fromty, toty, val):
    """
    Convert an IntEnum member to its raw integer value.
    """
    value = _get_member_value(builder, val)
    return context.cast(builder, value, fromty.dtype, toty)


@lower_constant(types.EnumMember)
def enum_constant(context, builder, ty, pyval):
    """
    Return a LLVM constant representing enum member *pyval*.
    """
    consts = [
        context.get_constant_generic(builder, ty.ntype, pyval.name),
        context.get_constant_generic(builder, ty.dtype, pyval.value)
    ]
    return impl_ret_borrowed(
        context, builder, ty, cgutils.pack_struct(builder, consts),
    )


@lower_getattr_generic(types.EnumClass)
def enum_class_getattr(context, builder, ty, val, attr):
    """
    Return an enum member by attribute name.
    """
    member = getattr(ty.instance_class, attr)
    return enum_constant(context, builder, ty, member)


@lower_builtin('static_getitem', types.EnumClass, types.StringLiteral)
def enum_class_getitem(context, builder, sig, args):
    """
    Return an enum member by index name.
    """
    enum_cls_typ, idx = sig.args
    member = enum_cls_typ.instance_class[idx.literal_value]
    return enum_constant(context, builder, enum_cls_typ, member)
