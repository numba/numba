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
        if npdatetime.can_cast_timedelta_units(left, right):
            return signature(right, left, right)
        elif npdatetime.can_cast_timedelta_units(right, left):
            return signature(left, left, right)


@builtin
class TimedeltaBinOpAdd(TimedeltaBinOp):
    key = "+"

