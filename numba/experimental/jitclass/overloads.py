"""
Overloads for ClassInstanceType for built-in functions that call dunder methods
on an object.
"""
import operator
import sys

from numba.core.extending import overload
from numba.core.types import ClassInstanceType


def _get_args(nargs=1):
    return list("xyzabcdefg")[:nargs]


def _simple_template(*attrs, nargs=1):
    args = _get_args(nargs=nargs)
    arg0 = args[0]

    template = f"""
def handler({",".join(args)}):
    if not isinstance({arg0}, ClassInstanceType):
        return
"""
    for attr in attrs:
        assert isinstance(attr, str)
        template += f"""
    if "__{attr}__" in {arg0}.jit_methods:
        return lambda {",".join(args)}: {arg0}.__{attr}__({",".join(args[1:])})
"""
    return template


def _register_overload(func, template, glbls=None):
    """
    Register overload handler for func calling class attribute attr.
    """
    namespace = dict(ClassInstanceType=ClassInstanceType)
    if glbls:
        namespace.update(glbls)

    exec(template, namespace)
    handler = namespace["handler"]
    overload(func)(handler)


def register_simple_overload(func, *attrs, nargs=1):
    template = _simple_template(*attrs, nargs=nargs)
    _register_overload(func, template)


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


register_simple_overload(abs, "abs")
register_simple_overload(len, "len")

# Comparison operators.
# register_simple_overload(operator.eq, "eq", 2)
# register_simple_overload(operator.ne, "ne", 2)
register_simple_overload(operator.ge, "ge", nargs=2)
register_simple_overload(operator.gt, "gt", nargs=2)
register_simple_overload(operator.le, "le", nargs=2)
register_simple_overload(operator.lt, "lt", nargs=2)

# Arithmetic operators.
register_simple_overload(operator.add, "add", nargs=2)
register_simple_overload(operator.floordiv, "floordiv", nargs=2)
register_simple_overload(operator.lshift, "lshift", nargs=2)
register_simple_overload(operator.mod, "mod", nargs=2)
register_simple_overload(operator.mul, "mul", nargs=2)
register_simple_overload(operator.neg, "neg")
register_simple_overload(operator.pos, "pos")
register_simple_overload(operator.pow, "pow", nargs=2)
register_simple_overload(operator.rshift, "rshift", nargs=2)
register_simple_overload(operator.sub, "sub", nargs=2)
register_simple_overload(operator.truediv, "truediv", nargs=2)

# Inplace arithmetic operators.
register_simple_overload(operator.iadd, "iadd", "add", nargs=2)
register_simple_overload(operator.ifloordiv, "ifloordiv", "floordiv", nargs=2)
register_simple_overload(operator.ilshift, "ilshift", "lshift", nargs=2)
register_simple_overload(operator.imod, "imod", "mod", nargs=2)
register_simple_overload(operator.imul, "imul", "mul", nargs=2)
register_simple_overload(operator.ipow, "ipow", "pow", nargs=2)
register_simple_overload(operator.irshift, "irshift", "rshift", nargs=2)
register_simple_overload(operator.isub, "isub", "sub", nargs=2)
register_simple_overload(operator.itruediv, "itruediv", "truediv", nargs=2)

# Logical operators.
register_simple_overload(operator.and_, "and", nargs=2)
register_simple_overload(operator.or_, "or", nargs=2)
register_simple_overload(operator.xor, "xor", nargs=2)

# Inplace logical operators.
register_simple_overload(operator.iand, "iand", "and", nargs=2)
register_simple_overload(operator.ior, "ior", "or", nargs=2)
register_simple_overload(operator.ixor, "ixor", "xor", nargs=2)
