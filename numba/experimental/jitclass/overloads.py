"""
Overloads for ClassInstanceType for built-in functions that call dunder methods
on an object.
"""
from functools import wraps
import inspect
import operator
import sys

from numba.core.extending import overload
from numba.core.types import ClassInstanceType


def _get_args(n_args):
    assert n_args in (1, 2)
    return list("xy")[:n_args]


def class_instance_overload(target):
    """
    Decorator to add an overload for target that applies when the first argument
    is a ClassInstanceType.
    """
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            if not isinstance(args[0], ClassInstanceType):
                return
            return func(*args, **kwargs)

        params = list(inspect.signature(wrapped).parameters)
        assert params == _get_args(len(params))
        return overload(target)(wrapped)

    return decorator


def extract_template(template, name):
    """
    Extract a code-generated function from a string template.
    """
    namespace = {}
    exec(template, namespace)
    return namespace[name]


def register_simple_overload(func, *attrs, n_args=1,):
    """
    Register an overload for func that checks for methods __attr__ for each
    attr in attrs.
    """
    # Use a template to set the signature correctly.
    arg_names = _get_args(n_args)
    template = f"""
def func({','.join(arg_names)}):
    pass
"""

    @wraps(extract_template(template, "func"))
    def overload_func(*args, **kwargs):
        options = [
            try_call_method(args[0], f"__{attr}__", n_args)
            for attr in attrs
        ]
        return take_first(*options)

    return class_instance_overload(func)(overload_func)


def try_call_method(cls_type, method, n_args=1):
    """
    If method is defined for cls_type, return a callable that calls this method.
    If not, return None.
    """
    if method in cls_type.jit_methods:
        arg_names = _get_args(n_args)
        template = f"""
def func({','.join(arg_names)}):
    return {arg_names[0]}.{method}({','.join(arg_names[1:])})
"""
        return extract_template(template, "func")


def take_first(*options):
    """
    Take the first non-None option.
    """
    assert all(o is None or inspect.isfunction(o) for o in options), options
    for o in options:
        if o is not None:
            return o


@class_instance_overload(bool)
def class_bool(x):
    return take_first(
        try_call_method(x, "__bool__"),
        try_call_method(x, "__len__"),
        lambda x: True,
    )


@class_instance_overload(complex)
def class_complex(x):
    return take_first(
        try_call_method(x, "__complex__"),
        lambda x: complex(float(x))
    )


@class_instance_overload(operator.contains)
def class_contains(x, y):
    # https://docs.python.org/3/reference/expressions.html#membership-test-operations
    return try_call_method(x, "__contains__", 2)
    # TODO: use __iter__ if defined.


@class_instance_overload(float)
def class_float(x):
    options = [try_call_method(x, "__float__")]

    if (
        (sys.version_info.major, sys.version_info.minor) >= (3, 8)
        and "__index__" in x.jit_methods
    ):
        options.append(lambda x: float(x.__index__()))

    return take_first(*options)


@class_instance_overload(int)
def class_int(x):
    options = [try_call_method(x, "__int__")]

    if (sys.version_info.major, sys.version_info.minor) >= (3, 8):
        options.append(try_call_method(x, "__index__"))

    return take_first(*options)


@class_instance_overload(str)
def class_str(x):
    return take_first(
        try_call_method(x, "__str__"),
        lambda x: repr(x),
    )


@class_instance_overload(operator.eq)
def class_eq(x, y):
    # TODO: Fallback to x is y.
    return try_call_method(x, "__eq__", 2)


@class_instance_overload(operator.ne)
def class_ne(x, y):
    return take_first(
        try_call_method(x, "__ne__", 2),
        lambda x, y: not (x == y),
    )


register_simple_overload(abs, "abs")
register_simple_overload(len, "len")

# Comparison operators.
register_simple_overload(hash, "hash")
register_simple_overload(operator.ge, "ge", n_args=2)
register_simple_overload(operator.gt, "gt", n_args=2)
register_simple_overload(operator.le, "le", n_args=2)
register_simple_overload(operator.lt, "lt", n_args=2)

# Arithmetic operators.
register_simple_overload(operator.add, "add", n_args=2)
register_simple_overload(operator.floordiv, "floordiv", n_args=2)
register_simple_overload(operator.lshift, "lshift", n_args=2)
register_simple_overload(operator.mul, "mul", n_args=2)
register_simple_overload(operator.mod, "mod", n_args=2)
register_simple_overload(operator.neg, "neg")
register_simple_overload(operator.pos, "pos")
register_simple_overload(operator.pow, "pow", n_args=2)
register_simple_overload(operator.rshift, "rshift", n_args=2)
register_simple_overload(operator.sub, "sub", n_args=2)
register_simple_overload(operator.truediv, "truediv", n_args=2)

# Inplace arithmetic operators.
register_simple_overload(operator.iadd, "iadd", "add", n_args=2)
register_simple_overload(operator.ifloordiv, "ifloordiv", "floordiv", n_args=2)
register_simple_overload(operator.ilshift, "ilshift", "lshift", n_args=2)
register_simple_overload(operator.imul, "imul", "mul", n_args=2)
register_simple_overload(operator.imod, "imod", "mod", n_args=2)
register_simple_overload(operator.ipow, "ipow", "pow", n_args=2)
register_simple_overload(operator.irshift, "irshift", "rshift", n_args=2)
register_simple_overload(operator.isub, "isub", "sub", n_args=2)
register_simple_overload(operator.itruediv, "itruediv", "truediv", n_args=2)

# Logical operators.
register_simple_overload(operator.and_, "and", n_args=2)
register_simple_overload(operator.or_, "or", n_args=2)
register_simple_overload(operator.xor, "xor", n_args=2)

# Inplace logical operators.
register_simple_overload(operator.iand, "iand", "and", n_args=2)
register_simple_overload(operator.ior, "ior", "or", n_args=2)
register_simple_overload(operator.ixor, "ixor", "xor", n_args=2)
