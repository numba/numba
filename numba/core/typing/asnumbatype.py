import inspect
import typing as py_typing

from numba.core.typing.typeof import typeof
from numba.core import errors, types, utils


def _py_version_switch(py_version, new_result, old_result):
    return new_result if utils.PYVERSION >= py_version else old_result


class AsNumbaTypeRegistry:
    """
    A registry for python typing declarations.  This registry stores a lookup
    table for simple cases (e.g. int) and a list of functions for more
    complicated cases (e.g. generics like List[int]).

    The as_numba_type registry is meant to work statically on type annotations
    at compile type, not dynamically on instances at runtime. To check the type
    of an object at runtime, see numba.typeof.
    """

    def __init__(self):
        self.lookup = {
            type(example): typeof(example)
            for example in [
                0,
                0.0,
                complex(0),
                "numba",
                True,
                None,
            ]
        }

        self.functions = [self._builtin_infer, self._numba_type_infer]

    def _numba_type_infer(self, py_type):
        if isinstance(py_type, types.Type):
            return py_type

    def _builtin_infer(self, py_type):
        # The type hierarchy of python typing library changes in 3.7.
        generic_type_check = _py_version_switch(
            (3, 7),
            lambda x: isinstance(x, py_typing._GenericAlias),
            lambda _: True,
        )
        if not generic_type_check(py_type):
            return

        list_origin = _py_version_switch((3, 7), list, py_typing.List)
        dict_origin = _py_version_switch((3, 7), dict, py_typing.Dict)
        set_origin = _py_version_switch((3, 7), set, py_typing.Set)
        tuple_origin = _py_version_switch((3, 7), tuple, py_typing.Tuple)

        if getattr(py_type, "__origin__", None) is py_typing.Union:
            if len(py_type.__args__) != 2:
                raise errors.TypingError(
                    "Cannot type Union of more than two types")

            (arg_1_py, arg_2_py) = py_type.__args__

            if arg_2_py is type(None): # noqa: E721
                return types.Optional(self.infer(arg_1_py))
            elif arg_1_py is type(None): # noqa: E721
                return types.Optional(self.infer(arg_2_py))
            else:
                raise errors.TypingError(
                    "Cannot type Union that is not an Optional "
                    f"(neither type type {arg_2_py} is not NoneType")

        if getattr(py_type, "__origin__", None) is list_origin:
            (element_py,) = py_type.__args__
            return types.ListType(self.infer(element_py))

        if getattr(py_type, "__origin__", None) is dict_origin:
            key_py, value_py = py_type.__args__
            return types.DictType(self.infer(key_py), self.infer(value_py))

        if getattr(py_type, "__origin__", None) is set_origin:
            (element_py,) = py_type.__args__
            return types.Set(self.infer(element_py))

        if getattr(py_type, "__origin__", None) is tuple_origin:
            tys = tuple(map(self.infer, py_type.__args__))
            return types.BaseTuple.from_types(tys)

    def register(self, func_or_py_type, numba_type=None):
        """
        Extend AsNumbaType to support new python types (e.g. a user defined
        JitClass).  For a simple pair of a python type and a numba type, can
        use as a function register(py_type, numba_type).  If more complex logic
        is required (e.g. for generic types), register can also be used as a
        decorator for a function that takes a python type as input and returns
        a numba type or None.
        """
        if numba_type is not None:
            # register used with a specific (py_type, numba_type) pair.
            assert isinstance(numba_type, types.Type)
            self.lookup[func_or_py_type] = numba_type
        else:
            # register used as a decorator.
            assert inspect.isfunction(func_or_py_type)
            self.functions.append(func_or_py_type)

    def try_infer(self, py_type):
        """
        Try to determine the numba type of a given python type.
        We first consider the lookup dictionary.  If py_type is not there, we
        iterate through the registered functions until one returns a numba type.
        If type inference fails, return None.
        """
        result = self.lookup.get(py_type, None)

        for func in self.functions:
            if result is not None:
                break
            result = func(py_type)

        if result is not None and not isinstance(result, types.Type):
            raise errors.TypingError(
                f"as_numba_type should return a numba type, got {result}"
            )
        return result

    def infer(self, py_type):
        result = self.try_infer(py_type)
        if result is None:
            raise errors.TypingError(
                f"Cannot infer numba type of python type {py_type}"
            )
        return result

    def __call__(self, py_type):
        return self.infer(py_type)


as_numba_type = AsNumbaTypeRegistry()
