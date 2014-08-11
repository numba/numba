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


class TimedeltaCmpOp(AbstractTemplate):

    def generic(self, args, kws):
        # For equality comparisons, all units are inter-comparable
        left, right = args
        if not all(isinstance(tp, types.NPTimedelta) for tp in args):
            return
        return signature(types.boolean, left, right)


class TimedeltaOrderedCmpOp(AbstractTemplate):

    def generic(self, args, kws):
        # For ordered comparisons, units must be compatible
        left, right = args
        if not all(isinstance(tp, types.NPTimedelta) for tp in args):
            return
        if (npdatetime.can_cast_timedelta_units(left.unit, right.unit) or
            npdatetime.can_cast_timedelta_units(right.unit, left.unit)):
            return signature(types.boolean, left, right)


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
        (timedelta64, timedelta64) -> float
        """
        left, right = args
        if not isinstance(left, types.NPTimedelta):
            return
        if isinstance(right, types.NPTimedelta):
            if (npdatetime.can_cast_timedelta_units(left.unit, right.unit)
                or npdatetime.can_cast_timedelta_units(right.unit, left.unit)):
                return signature(types.float64, left, right)
        # Force integer types to convert to signed because it matches
        # timedelta64 semantics better.
        elif right in types.signed_domain or right in types.real_domain:
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
class TimedeltaTrueDiv(TimedeltaDivOp):
    key = "/"

@builtin
class TimedeltaFloorDiv(TimedeltaDivOp):
    key = "//"

@builtin
class TimedeltaLegacyDiv(TimedeltaDivOp):
    key = "/?"

@builtin
class TimedeltaCmpEq(TimedeltaCmpOp):
    key = '=='

@builtin
class TimedeltaCmpNe(TimedeltaCmpOp):
    key = '!='

@builtin
class TimedeltaCmpLt(TimedeltaOrderedCmpOp):
    key = '<'

@builtin
class TimedeltaCmpLE(TimedeltaOrderedCmpOp):
    key = '<='

@builtin
class TimedeltaCmpGt(TimedeltaOrderedCmpOp):
    key = '>'

@builtin
class TimedeltaCmpGE(TimedeltaOrderedCmpOp):
    key = '>='
