import unittest
from numba import njit
from numba.core.base import BaseContext
from numba.core.extending import overload_attribute
from numba.extending import (
    overload_method,
    models,
    register_model,
    types,
    intrinsic,
    typeof_impl,
    type_callable,
    lower_builtin,
)
from numba.core import cgutils
from numba.tests.support import TestCase


class Foo:
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return self.a + x


class FooType(types.Dummy):
    """
    Type class associated with the `cublasdx.BlasNumba`.
    """

    def __init__(self):
        self.name = "FooType"


foo_type = FooType()


# Register type for Foo
@typeof_impl.register(type)
def _typeof_type(val, c):
    """
    Type various specific Python types.
    """
    # Same as typed.Dict, typed.List
    if issubclass(val, Foo):
        return types.TypeRef(FooType)


# Register model for Foo
@register_model(FooType)
class FooStructModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("a", types.int64)]
        super().__init__(dmm, fe_type, members)


# Teaching numba that we can call `Foo()` - typing
@type_callable(FooType)
def _type_callable_Foo(context):
    def typer(a):
        if not isinstance(a, types.Integer):
            return
        return foo_type

    return typer


# Teaching numba that we can call `Foo()` - implementation
@lower_builtin(types.TypeRef(FooType), types.Integer)
def lower_foo_type_ctor(context, builder, sig, args):
    obj = cgutils.create_struct_proxy(foo_type)(context, builder)
    obj.a = args[0]
    return obj._getvalue()


@intrinsic
def foo_attr_a(typingctx, foo_ty):
    def impl(context: BaseContext, builder, signature, args):
        obj = cgutils.create_struct_proxy(foo_type)(
            context, builder, value=args[0])

        return obj.a

    return types.int64(foo_type), impl


@overload_attribute(FooType, "a")
def ol_Foo_a(self):
    return lambda self: foo_attr_a(self)


@overload_method(FooType, "__call__", strict=None)
def ol_Foo___call__(self, x):
    return lambda self, x: self.a + x


@njit
def use_method_overload(a, b):
    foo = Foo(a)

    return foo(b)


class TestOverload(TestCase):
    def test_check_overload(self):
        assert use_method_overload(1, 2) == 3


if __name__ == "__main__":
    unittest.main()
