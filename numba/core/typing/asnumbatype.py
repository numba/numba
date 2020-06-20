import inspect
import typing as py_typing

from numba.core.typing.typeof import typeof
from numba.core import errors, types, utils


class AsNumbaTypeRegistry:
    """
    A registry for python typing declarations.  This registry stores functions
    that take a Python typing type (e.g. int or List[float]) and return None or
    a numba type.
    """

    def __init__(self):
        self.functions = []

    def register(self, func):
        assert inspect.isfunction(func)
        self.functions.append(func)

    def infer(self, py_type):
        for func in self.functions:
            result = func(py_type)
            if result is not None:
                return result

        raise errors.TypingError(
            f"Cannot infer numba type of python type {py_type}"
        )


as_numba_type_registry = AsNumbaTypeRegistry()

register_py_type_infer = as_numba_type_registry.register
as_numba_type = as_numba_type_registry.infer


def python_switch(py_version, new_result, old_result):
    return new_result if utils.PYVERSION >= py_version else old_result


@register_py_type_infer
def builtin_typer(py_type):
    print("builtin_typer", py_type)
    if py_type in (int, float, complex):
        return typeof(py_type(0))

    if py_type is str:
        return typeof("numba")

    # The type hierarchy of python typing library changes in 3.7.
    list_origin = python_switch((3, 7), list, py_typing.List)
    dict_origin = python_switch((3, 7), dict, py_typing.Dict)
    set_origin = python_switch((3, 7), set, py_typing.Set)
    tuple_origin = python_switch((3, 7), tuple, py_typing.Tuple)

    generic_type_check = python_switch(
        (3, 7),
        lambda x: isinstance(x, py_typing._GenericAlias),
        lambda _: True,
    )

    if generic_type_check(py_type) and getattr(py_type, "__origin__", None) is py_typing.Union:
        (arg_1_py, arg_2_py) = py_type.__args__
        if arg_2_py is not type(None): # noqa: E721
            raise errors.TypingError(
                "Cannot type Union that is not an Optional "
                f"(second type {arg_2_py} is not NoneType")
        return types.Optional(as_numba_type(arg_1_py))

    if generic_type_check(py_type) and getattr(py_type, "__origin__", None) is list_origin:
        (element_py,) = py_type.__args__
        return types.List(as_numba_type(element_py))

    if generic_type_check(py_type) and getattr(py_type, "__origin__", None) is dict_origin:
        key_py, value_py = py_type.__args__
        return types.DictType(as_numba_type(key_py), as_numba_type(value_py))

    if generic_type_check(py_type) and getattr(py_type, "__origin__", None) is set_origin:
        (element_py,) = py_type.__args__
        return types.Set(as_numba_type(element_py))

    if generic_type_check(py_type) and getattr(py_type, "__origin__", None) is tuple_origin:
        tys = tuple(map(as_numba_type, py_type.__args__))
        return types.BaseTuple.from_types(tys)
