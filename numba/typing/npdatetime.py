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
        """
        left, right = args
        if isinstance(right, types.NPTimedelta):
            td, other = right, left
        elif isinstance(left, types.NPTimedelta):
            td, other = left, right
        if not isinstance(other, (types.Integer, types.Float)):
            return
        return signature(td, left, right)


@builtin
class TimedeltaBinAdd(TimedeltaBinOp):
    key = "+"

@builtin
class TimedeltaBinSub(TimedeltaBinOp):
    key = "-"

@builtin
class TimedeltaBinMult(TimedeltaMixOp):
    key = "*"

