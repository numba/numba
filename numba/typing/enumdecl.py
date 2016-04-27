"""
Typing for enums.
"""

import enum

from numba import types, utils
from numba.typing.templates import (AbstractTemplate, AttributeTemplate,
                                    signature, Registry, bound_function)

registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr


@infer_getattr
class EnumAttribute(AttributeTemplate):
    key = types.EnumMember

    def resolve_value(self, ty):
        return ty.dtype


@infer_getattr
class EnumClassAttribute(AttributeTemplate):
    key = types.EnumClass

    def generic_resolve(self, ty, attr):
        """
        Resolve attributes of an enum class as enum members.
        """
        if attr in ty.instance_class.__members__:
            return ty.member_type


class EnumCompare(AbstractTemplate):

    def generic(self, args, kws):
        [lhs, rhs] = args
        if (isinstance(lhs, types.EnumMember)
            and isinstance(rhs, types.EnumMember)
            and lhs == rhs):
            return signature(types.boolean, lhs, rhs)

@infer
class EnumEq(EnumCompare):
    key = '=='

@infer
class EnumNe(EnumCompare):
    key = '!='
