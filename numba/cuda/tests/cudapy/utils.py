from numba import types, float32
from numba.core.extending import (
    models,
    register_model,
    make_attribute_wrapper,
    typeof_impl,
    type_callable
)
from numba.cuda.cudaimpl import lower
from numba.core import cgutils


class TestStruct:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class TestStructModelType(types.Type):
    def __init__(self):
        super().__init__(name="TestStructModelType")


test_struct_model_type = TestStructModelType()


@typeof_impl.register(TestStruct)
def typeof_teststruct(val, c):
    return test_struct_model_type


@register_model(TestStructModelType)
class TestStructModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("x", float32), ("y", float32)]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(TestStructModelType, 'x', 'x')
make_attribute_wrapper(TestStructModelType, 'y', 'y')


@type_callable(TestStruct)
def type_test_struct(context):
    def typer(x, y):
        if isinstance(x, types.Float) and isinstance(y, types.Float):
            return test_struct_model_type
    return typer


@lower(TestStruct, types.Float, types.Float)
def lower_test_type_ctor(context, builder, sig, args):
    obj = cgutils.create_struct_proxy(test_struct_model_type)(context, builder)
    obj.x = args[0]
    obj.y = args[1]
    return obj._getvalue()
