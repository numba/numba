"""
Overloads for ClassInstanceType for built-in functions that call dunder methods
on an object.
"""
import operator
import sys

from numba.core.extending import overload
from numba.core.types import ClassInstanceType


def register_class_overload(func, attr, nargs=1):
    """
    Register overload handler for func calling class attribute attr.
    """
    args = list("abcdefg")[:nargs]
    arg0 = args[0]

    template = f"""
def handler({",".join(args)}):
    if not isinstance({arg0}, ClassInstanceType):
        return
    if "__{attr}__" in {arg0}.jit_methods:
        return lambda {",".join(args)}: {arg0}.__{attr}__({",".join(args[1:])})
"""

    namespace = dict(ClassInstanceType=ClassInstanceType)
    exec(template, namespace)

    handler = namespace["handler"]
    overload(func)(handler)


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


@overload(str)
def class_str(x):
    if not isinstance(x, ClassInstanceType):
        return

    if "__str__" in x.jit_methods:
        return lambda x: x.__str__()

    return lambda x: repr(x)


register_class_overload(abs, "abs")
register_class_overload(len, "len")

# Comparison operators.
# register_class_overload(operator.eq, "eq", 2)
# register_class_overload(operator.ne, "ne", 2)
register_class_overload(operator.ge, "ge", 2)
register_class_overload(operator.gt, "gt", 2)
register_class_overload(operator.le, "le", 2)
register_class_overload(operator.lt, "lt", 2)

# Arithmetic operators.
register_class_overload(operator.add, "add", 2)
register_class_overload(operator.floordiv, "floordiv", 2)
register_class_overload(operator.lshift, "lshift", 2)
register_class_overload(operator.mod, "mod", 2)
register_class_overload(operator.mul, "mul", 2)
register_class_overload(operator.neg, "neg")
register_class_overload(operator.pos, "pos")
register_class_overload(operator.pow, "pow", 2)
register_class_overload(operator.rshift, "rshift", 2)
register_class_overload(operator.sub, "sub", 2)
register_class_overload(operator.truediv, "truediv", 2)

# Logical operators.
register_class_overload(operator.and_, "and", 2)
register_class_overload(operator.or_, "or", 2)
register_class_overload(operator.xor, "xor", 2)
