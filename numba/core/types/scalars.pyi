import functools
from typing import ClassVar, Final, SupportsComplex, SupportsFloat, SupportsIndex, override

import numpy as np
from typing_extensions import Generic, Self, TypeVar

from .abstract import Dummy, Hashable, Literal, Number, Type

_NumpyIntT_co = TypeVar(
    "_NumpyIntT_co",
    bound=np.integer,
    default=np.integer,
    covariant=True,
)
_NumpyFloatT_co = TypeVar(
    "_NumpyFloatT_co",
    bound=np.floating,
    default=np.floating,
    covariant=True,
)
_NumpyComplexT_co = TypeVar(
    "_NumpyComplexT_co",
    bound=np.complexfloating,
    default=np.complexfloating,
    covariant=True,
)

###

def parse_integer_bitwidth(name: str) -> int: ...
def parse_integer_signed(name: str) -> bool: ...

class Boolean(Hashable):
    @override
    def cast_python_value(self, value: object) -> bool: ...

class BooleanLiteral(Literal[bool], Boolean):
    @override
    def __init__(self, value: bool) -> None: ...

@functools.total_ordering
class Integer(Number, Generic[_NumpyIntT_co]):
    @classmethod
    def from_bitwidth(cls, bitwidth: int, signed: bool = True) -> Self: ...

    #
    @override
    def __init__(
        self,
        name: str,
        bitwidth: int | None = None,
        signed: bool | None = None,
    ) -> None: ...
    @override
    def cast_python_value(self, value: SupportsIndex) -> _NumpyIntT_co: ...
    def __lt__(self, other: Self, /) -> bool: ...

    #
    @property
    def maxval(self) -> int: ...
    @property
    def minval(self) -> int: ...

class IntegerLiteral(Literal[int], Integer[_NumpyIntT_co], Generic[_NumpyIntT_co]):
    @override
    def __init__(self, value: int) -> None: ...

@functools.total_ordering
class Float(Number, Generic[_NumpyFloatT_co]):
    bitwidth: Final[int]

    @override
    def __init__(self, name: str) -> None: ...
    @override
    def cast_python_value(
        self,
        value: SupportsFloat | SupportsIndex,
    ) -> _NumpyFloatT_co: ...
    def __lt__(self, other: Self, /) -> bool: ...

@functools.total_ordering
class Complex(Number, Generic[_NumpyComplexT_co, _NumpyFloatT_co]):
    bitwidth: Final[int]
    underlying_float: Float[_NumpyFloatT_co]

    @override
    def __init__(self, name: str, underlying_float: Float[_NumpyFloatT_co]) -> None: ...
    @override
    def cast_python_value(
        self,
        value: SupportsComplex | SupportsFloat | SupportsIndex,
    ) -> _NumpyComplexT_co: ...
    def __lt__(self, other: Self, /) -> bool: ...

class EnumClass(Dummy):
    basename: ClassVar[str] = "Enum class"

    instance_class: Final[type]
    dtype: Final[Type]

    @override
    def __init__(self, cls: type, dtype: Type) -> None: ...

    #
    @property
    @override
    def key(self) -> tuple[type, Type]: ...
    @functools.cached_property
    def member_type(self) -> EnumMember: ...

class IntEnumClass(EnumClass):
    basename: ClassVar[str] = "IntEnum class"

    @functools.cached_property
    @override
    def member_type(self) -> IntEnumMember: ...

class EnumMember(Type):
    basename: ClassVar[str] = "Enum"
    class_type_class: ClassVar[type[EnumClass]] = EnumClass

    instance_class: Final[type]
    dtype: Final[Type]

    @override
    def __init__(self, cls: type, dtype: Type) -> None: ...

    #
    @property
    @override
    def key(self) -> tuple[type, Type]: ...
    @property
    def class_type(self) -> EnumClass: ...

class IntEnumMember(EnumMember):
    basename: ClassVar[str] = "IntEnum"
    class_type_class: ClassVar[type[IntEnumClass]] = IntEnumClass
