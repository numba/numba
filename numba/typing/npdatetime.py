"""
Typing declarations for numpy.timedelta64.
"""

from __future__ import print_function, division, absolute_import

from itertools import product

from numba import npdatetime, types
from numba.utils import PYVERSION
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    AbstractTemplate, builtin_global, builtin,
                                    builtin_attr, signature)


class TimedeltaBinOp(AbstractTemplate):

    def generic(self, args, kws):
        left, right = args
        if not all(isinstance(tp, types.NPTimedelta) for tp in args):
            return
        if npdatetime.can_cast_timedelta_units(left.unit, right.unit):
            return signature(right, left, right)
        elif npdatetime.can_cast_timedelta_units(right.unit, left.unit):
            return signature(left, left, right)


class TimedeltaMixOp(AbstractTemplate):

    def generic(self, args, kws):
        """
        (timedelta64, {int, float}) -> timedelta64
        ({int, float}, timedelta64) -> timedelta64
        """
        left, right = args
        if isinstance(right, types.NPTimedelta):
            td, other = right, left
        elif isinstance(left, types.NPTimedelta):
            td, other = left, right
        # Force integer types to convert to signed because it matches
        # timedelta64 semantics better.
        if other not in types.signed_domain and other not in types.real_domain:
            return
        return signature(td, left, right)


class TimedeltaDivOp(AbstractTemplate):

    def generic(self, args, kws):
        """
        (timedelta64, {int, float}) -> timedelta64
        """
        left, right = args
        # Force integer types to convert to signed because it matches
        # timedelta64 semantics better.
        if right not in types.signed_domain and right not in types.real_domain:
            return
        if isinstance(left, types.NPTimedelta):
            return signature(left, left, right)


@builtin
class TimedeltaBinAdd(TimedeltaBinOp):
    key = "+"

@builtin
class TimedeltaBinSub(TimedeltaBinOp):
    key = "-"

@builtin
class TimedeltaBinMult(TimedeltaMixOp):
    key = "*"

@builtin
class TimedeltaDiv(TimedeltaDivOp):
    key = "/"

@builtin
class TimedeltaFloorDiv(TimedeltaDivOp):
    key = "//"

