"""
Typing declarations for np.timedelta64.
"""

from __future__ import print_function, division, absolute_import

from itertools import product

from numba import npdatetime, types
from numba.utils import PYVERSION
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    AbstractTemplate, infer_global, infer,
                                    infer_getattr, signature)


# timedelta64-only operations

class TimedeltaUnaryOp(AbstractTemplate):

    def generic(self, args, kws):
        if len(args) == 2:
            # Guard against binary + and -
            return
        op, = args
        if not isinstance(op, types.NPTimedelta):
            return
        return signature(op, op)


class TimedeltaBinOp(AbstractTemplate):

    def generic(self, args, kws):
        if len(args) == 1:
            # Guard against unary + and -
            return
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
            sig_factory = lambda other: signature(td, other, td)
        elif isinstance(left, types.NPTimedelta):
            td, other = left, right
            sig_factory = lambda other: signature(td, td, other)
        else:
            return
        if not isinstance(other, (types.Float, types.Integer)):
            return
        # Force integer types to convert to signed because it matches
        # timedelta64 semantics better.
        if isinstance(other, types.Integer):
            other = types.int64
        return sig_factory(other)


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
        elif isinstance(right, (types.Float)):
            return signature(left, left, right)
        elif isinstance(right, (types.Integer)):
            # Force integer types to convert to signed because it matches
            # timedelta64 semantics better.
            return signature(left, left, types.int64)


@infer
class TimedeltaUnaryPos(TimedeltaUnaryOp):
    key = "+"

@infer
class TimedeltaUnaryNeg(TimedeltaUnaryOp):
    key = "-"

@infer
class TimedeltaBinAdd(TimedeltaBinOp):
    key = "+"

@infer
class TimedeltaBinSub(TimedeltaBinOp):
    key = "-"

@infer
class TimedeltaBinMult(TimedeltaMixOp):
    key = "*"

@infer
class TimedeltaTrueDiv(TimedeltaDivOp):
    key = "/"

@infer
class TimedeltaFloorDiv(TimedeltaDivOp):
    key = "//"

@infer
class TimedeltaLegacyDiv(TimedeltaDivOp):
    key = "/?"

@infer
class TimedeltaCmpEq(TimedeltaCmpOp):
    key = '=='

@infer
class TimedeltaCmpNe(TimedeltaCmpOp):
    key = '!='

@infer
class TimedeltaCmpLt(TimedeltaOrderedCmpOp):
    key = '<'

@infer
class TimedeltaCmpLE(TimedeltaOrderedCmpOp):
    key = '<='

@infer
class TimedeltaCmpGt(TimedeltaOrderedCmpOp):
    key = '>'

@infer
class TimedeltaCmpGE(TimedeltaOrderedCmpOp):
    key = '>='


@infer_global(abs)
class TimedeltaAbs(TimedeltaUnaryOp):
    pass


# datetime64 operations

@infer
class DatetimePlusTimedelta(AbstractTemplate):
    key = '+'

    def generic(self, args, kws):
        if len(args) == 1:
            # Guard against unary +
            return
        left, right = args
        if isinstance(right, types.NPTimedelta):
            dt = left
            td = right
        elif isinstance(left, types.NPTimedelta):
            dt = right
            td = left
        else:
            return
        if isinstance(dt, types.NPDatetime):
            unit = npdatetime.combine_datetime_timedelta_units(dt.unit, td.unit)
            if unit is not None:
                return signature(types.NPDatetime(unit), left, right)

@infer
class DatetimeMinusTimedelta(AbstractTemplate):
    key = '-'

    def generic(self, args, kws):
        if len(args) == 1:
            # Guard against unary -
            return
        dt, td = args
        if isinstance(dt, types.NPDatetime) and isinstance(td, types.NPTimedelta):
            unit = npdatetime.combine_datetime_timedelta_units(dt.unit, td.unit)
            if unit is not None:
                return signature(types.NPDatetime(unit), dt, td)

@infer
class DatetimeMinusDatetime(AbstractTemplate):
    key = '-'

    def generic(self, args, kws):
        if len(args) == 1:
            # Guard against unary -
            return
        left, right = args
        if isinstance(left, types.NPDatetime) and isinstance(right, types.NPDatetime):
            # All units compatible! Yoohoo!
            unit = npdatetime.get_best_unit(left.unit, right.unit)
            return signature(types.NPTimedelta(unit), left, right)


class DatetimeCmpOp(AbstractTemplate):

    def generic(self, args, kws):
        # For datetime64 comparisons, all units are inter-comparable
        left, right = args
        if not all(isinstance(tp, types.NPDatetime) for tp in args):
            return
        return signature(types.boolean, left, right)


@infer
class DatetimeCmpEq(DatetimeCmpOp):
    key = '=='

@infer
class DatetimeCmpNe(DatetimeCmpOp):
    key = '!='

@infer
class DatetimeCmpLt(DatetimeCmpOp):
    key = '<'

@infer
class DatetimeCmpLE(DatetimeCmpOp):
    key = '<='

@infer
class DatetimeCmpGt(DatetimeCmpOp):
    key = '>'

@infer
class DatetimeCmpGE(DatetimeCmpOp):
    key = '>='
