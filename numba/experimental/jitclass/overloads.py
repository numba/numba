"""
Overloads for ClassInstanceType for built-in functions that call dunder methods
on an object.
"""
import operator
import sys

from numba.core.extending import overload
from numba.core.types import ClassInstanceType


@overload(abs)
def class_abs(x):
    if not isinstance(x, ClassInstanceType):
        return

    if "__abs__" in x.jit_methods:
        return lambda x: x.__abs__()


@overload(bool)
def class_bool(x):
    if not isinstance(x, ClassInstanceType):
        return

    if "__bool__" in x.jit_methods:
        return lambda x: x.__bool__()

    if "__len__" in x.jit_methods:
        return lambda x: x.__len__() != 0

    return lambda x: True


@overload(complex)
def class_complex(x):
    if not isinstance(x, ClassInstanceType):
        return

    if "__complex__" in x.jit_methods:
        return lambda x: x.__complex__()

    return lambda x: complex(float(x))


@overload(operator.contains)
def class_contains(x, y):
    # https://docs.python.org/3/reference/expressions.html#membership-test-operations
    if not isinstance(x, ClassInstanceType):
        return

    if "__contains__" in x.jit_methods:
        return lambda x, y: x.__contains__(y)

    # TODO: use __iter__ if defined.


@overload(float)
def class_float(x):
    if not isinstance(x, ClassInstanceType):
        return

    if "__float__" in x.jit_methods:
        return lambda x: x.__float__()

    if ((sys.version_info.major, sys.version_info.minor) >= (3, 8) and
            "__index__" in x.jit_methods):
        return lambda x: float(x.__index__())


@overload(int)
def class_int(x):
    if not isinstance(x, ClassInstanceType):
        return

    if "__int__" in x.jit_methods:
        return lambda x: x.__int__()

    if ((sys.version_info.major, sys.version_info.minor) >= (3, 8) and
            "__index__" in x.jit_methods):
        return lambda x: x.__index__()


@overload(len)
def class_len(x):
    if not isinstance(x, ClassInstanceType):
        return

    if "__len__" in x.jit_methods:
        return lambda x: x.__len__()
