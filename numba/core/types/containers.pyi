from collections.abc import Iterable as _PyIterable
from collections.abc import Iterator as _PyIterator
from collections.abc import Mapping as _PyMapping
from collections.abc import Sequence as _PySequence
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Generic,
    Protocol,
    SupportsIndex,
    TypeAlias,
    overload,
    type_check_only,
)
from typing import Literal as _PyLiteral

from _typeshed import Unused
from typing_extensions import Self, TypeVar, TypeVarTuple, Unpack, override

from .abstract import (
    ConstSized,
    Container,
    Hashable,
    InitialValue,
    IterableType,
    Literal,
    MutableSequence,
    Poison,
    Sequence,
    Type,
)
from .common import Buffer, SimpleIterableType, SimpleIteratorType

###
# ruff: noqa:PLR2044

_T = TypeVar("_T")
_InitT_co = TypeVar("_InitT_co", covariant=True, default=Any | None)
_TypeT = TypeVar("_TypeT", bound=Type)
_TypeT1 = TypeVar("_TypeT1", bound=Type)
_TypeT2 = TypeVar("_TypeT2", bound=Type)
_TypeT3 = TypeVar("_TypeT3", bound=Type)
_TypeT4 = TypeVar("_TypeT4", bound=Type)
_TypeT5 = TypeVar("_TypeT5", bound=Type)
_TypeTs = TypeVarTuple("_TypeTs", default=Unpack[tuple[Any, ...]])
_TypeT_co = TypeVar("_TypeT_co", bound=Type, default=Type, covariant=True)
_TypeT_contra = TypeVar("_TypeT_contra", bound=Type, default=Type, contravariant=True)
_TypeT1_co = TypeVar("_TypeT1_co", bound=Type, default=Type, covariant=True)
_TypeT2_co = TypeVar("_TypeT2_co", bound=Type, default=Type, covariant=True)
_KeyTypeT_co = TypeVar("_KeyTypeT_co", bound=Type, default=Type, covariant=True)
_ValTypeT_co = TypeVar("_ValTypeT_co", bound=Type, default=Type, covariant=True)

_ContainerT_co = TypeVar(
    "_ContainerT_co",
    bound=Container,
    default=Container,
    covariant=True,
)
_ListT_co = TypeVar(
    "_ListT_co",
    bound=List[Any, Any],
    default=List[Any, Any],
    covariant=True,
)
_SetT_co = TypeVar("_SetT_co", bound=Set[Any], default=Set[Any], covariant=True)

###

class Pair(Type, Generic[_TypeT1_co, _TypeT2_co]):
    first_type: _TypeT1_co
    second_type: _TypeT2_co

    def __init__(self, first_type: _TypeT1_co, second_type: _TypeT2_co) -> None: ...

    #
    @property
    @override
    def key(self) -> tuple[_TypeT1_co, _TypeT2_co]: ...

class BaseContainerIterator(SimpleIteratorType[_TypeT_co], Generic[_TypeT_co]):
    container_class: ClassVar[type[Container]]  # associated type
    container: Container[_TypeT_co]  # instance of `container_class`

    def __init__(self, container: Container[_TypeT_co]) -> None: ...

    #
    @property
    @override
    def key(self) -> Container[_TypeT_co]: ...

class BaseContainerPayload(Type, Generic[_ContainerT_co]):
    container_class: ClassVar[type[Container]]  # associated type
    container: _ContainerT_co  # instance of `container_class`

    def __init__(self, container: _ContainerT_co) -> None: ...

    #
    @property
    @override
    def key(self) -> _ContainerT_co: ...

class Bytes(Buffer):
    slice_is_copy: ClassVar[bool] = False
    mutable: bool = False

class ByteArray(Buffer):
    slice_is_copy: ClassVar[bool] = True

class PyArray(Buffer):
    slice_is_copy: ClassVar[bool] = True

class MemoryView(Buffer): ...

class BaseTuple(ConstSized, Hashable):
    @classmethod
    def from_types(
        cls,
        tys: _PySequence[Type],
        pyclass: type[tuple[Any, ...]] | None = None,
    ) -> BaseTuple: ...

class BaseAnonymousTuple(BaseTuple):
    def __unliteral__(self) -> BaseAnonymousTuple: ...

class _HomogeneousTuple(Sequence[_TypeT_co], BaseTuple, Generic[_TypeT_co]):
    @property
    @override
    def iterator_type(self) -> UniTupleIter[_TypeT_co]: ...
    @property
    def types(self) -> tuple[_TypeT_co, ...]: ...

    #
    @override
    def __getitem__(self, i: Unused) -> _TypeT_co: ...  # pyrefly:ignore[bad-override]
    def __iter__(self) -> _PyIterator[_TypeT_co]: ...
    @override
    def __len__(self) -> int: ...

class UniTuple(BaseAnonymousTuple, _HomogeneousTuple[_TypeT_co]):
    dtype: _TypeT_co
    count: int

    def __init__(self, dtype: _TypeT_co, count: int) -> None: ...
    @override
    def __unliteral__(self) -> UniTuple[Any]: ...

    #
    @property
    @override
    def mangling_args(self) -> tuple[str, tuple[_TypeT_co, int]]: ...
    @property
    @override
    def key(self) -> tuple[_TypeT_co, int]: ...

class UniTupleIter(BaseContainerIterator[_TypeT_co], Generic[_TypeT_co]):
    container_class: ClassVar[type[_HomogeneousTuple]] = ...

class _HeterogeneousTuple(BaseTuple, Generic[Unpack[_TypeTs]]):
    @staticmethod
    def is_types_iterable(types: _PyIterable[object]) -> None: ...

    # verified to work as intended for i <= 5
    @override
    @overload
    def __getitem__(  # pyrefly:ignore[bad-override]
        self: _HeterogeneousTuple[_T, Unpack[tuple[Any, ...]]],
        i: _PyLiteral[0],
    ) -> _T: ...
    @overload
    def __getitem__(
        self: _HeterogeneousTuple[Any, _T, Unpack[tuple[Any, ...]]],
        i: _PyLiteral[1],
    ) -> _T: ...
    @overload
    def __getitem__(
        self: _HeterogeneousTuple[Any, Any, _T, Unpack[tuple[Any, ...]]],
        i: _PyLiteral[2],
    ) -> _T: ...
    @overload
    def __getitem__(
        self: _HeterogeneousTuple[Any, Any, Any, _T, Unpack[tuple[Any, ...]]],
        i: _PyLiteral[3],
    ) -> _T: ...
    @overload
    def __getitem__(
        self: _HeterogeneousTuple[Any, Any, Any, Any, _T, Unpack[tuple[Any, ...]]],
        i: _PyLiteral[4],
    ) -> _T: ...
    @overload
    def __getitem__(
        self: _HeterogeneousTuple[Any, Any, Any, Any, Any, _T, Unpack[tuple[Any, ...]]],
        i: _PyLiteral[5],
    ) -> _T: ...
    @overload
    def __getitem__(
        self: _HeterogeneousTuple[Unpack[tuple[_T, ...]]],
        i: SupportsIndex,
    ) -> _T: ...

    #
    @override
    def __len__(self) -> int: ...
    def __iter__(
        self: _HeterogeneousTuple[Unpack[tuple[_T, ...]]],
    ) -> _PyIterator[_T]: ...

class UnionType(Type, Generic[_TypeT_contra]):
    types: tuple[_TypeT_contra, ...]

    def __init__(self, types: _PyIterable[_TypeT_contra]) -> None: ...
    def get_type_tag(self, typ: _TypeT_contra) -> int: ...

class Tuple(
    BaseAnonymousTuple,
    _HeterogeneousTuple[Unpack[_TypeTs]],
    Generic[Unpack[_TypeTs]],
):
    types: tuple[Unpack[_TypeTs]]
    count: int
    dtype: UnionType[Any]  # it's currently not possible to `typing.Union[Unpack[_Ts]]`

    # We cannot enforce TypeVar equality in Python's type system, so it's not possible
    # to distinguish between homogeneous/heterogeneous tuples (when n>1).
    @overload
    def __new__(cls, types: tuple[()]) -> Tuple[Unpack[tuple[()]]]: ...
    @overload
    def __new__(cls, types: tuple[_TypeT]) -> UniTuple[_TypeT]: ...
    @overload
    def __new__(
        cls,
        types: tuple[_TypeT1, _TypeT2],
    ) -> UniTuple[_TypeT1 | _TypeT2] | Tuple[_TypeT1, _TypeT2]: ...
    @overload
    def __new__(
        cls,
        types: tuple[_TypeT1, _TypeT2, _TypeT3],
    ) -> UniTuple[_TypeT1 | _TypeT2 | _TypeT3] | Tuple[_TypeT1, _TypeT2, _TypeT3]: ...
    @overload
    def __new__(
        cls,
        types: tuple[_TypeT1, _TypeT2, _TypeT3, _TypeT4],
    ) -> (
        UniTuple[_TypeT1 | _TypeT2 | _TypeT3 | _TypeT4]
        | Tuple[_TypeT1, _TypeT2, _TypeT3, _TypeT4]
    ): ...
    @overload
    def __new__(
        cls,
        types: tuple[_TypeT1, _TypeT2, _TypeT3, _TypeT4, _TypeT5],
    ) -> (
        UniTuple[_TypeT1 | _TypeT2 | _TypeT3 | _TypeT4 | _TypeT5]
        | Tuple[_TypeT1, _TypeT2, _TypeT3, _TypeT4, _TypeT5]
    ): ...
    @overload
    def __new__(
        cls,
        types: _PyIterable[_TypeT],
    ) -> UniTuple[_TypeT] | Tuple[Unpack[tuple[_TypeT, ...]]]: ...

    #
    @overload
    def __init__(self, types: tuple[Unpack[_TypeTs]]) -> None: ...
    @overload
    def __init__(
        self: Tuple[Unpack[tuple[_T, ...]]],
        types: _PyIterable[_T],
    ) -> None: ...

    #
    @property
    @override
    def mangling_args(self) -> tuple[str, tuple[Unpack[_TypeTs]]]: ...
    @property
    @override
    def key(self) -> tuple[Unpack[_TypeTs]]: ...

class _StarArgTupleMixin: ...

class StarArgTuple(
    _StarArgTupleMixin,
    Tuple[Unpack[_TypeTs]],
    Generic[Unpack[_TypeTs]],
):
    # Python's type system has no support for higher kinded typing, so we're forced to
    # duplicate the `Tuple.__new__` overloads, changing only the return _kinds_.
    # See https://github.com/python/typing/issues/548 for details.
    @overload
    def __new__(cls, types: tuple[()]) -> StarArgTuple[Unpack[tuple[()]]]: ...
    @overload
    def __new__(cls, types: tuple[_TypeT]) -> StarArgUniTuple[_TypeT]: ...
    @overload
    def __new__(
        cls,
        types: tuple[_TypeT1, _TypeT2],
    ) -> StarArgUniTuple[_TypeT1 | _TypeT2] | StarArgTuple[_TypeT1, _TypeT2]: ...
    @overload
    def __new__(
        cls,
        types: tuple[_TypeT1, _TypeT2, _TypeT3],
    ) -> (
        StarArgUniTuple[_TypeT1 | _TypeT2 | _TypeT3]
        | StarArgTuple[_TypeT1, _TypeT2, _TypeT3]
    ): ...
    @overload
    def __new__(
        cls,
        types: tuple[_TypeT1, _TypeT2, _TypeT3, _TypeT4],
    ) -> (
        StarArgUniTuple[_TypeT1 | _TypeT2 | _TypeT3 | _TypeT4]
        | StarArgTuple[_TypeT1, _TypeT2, _TypeT3, _TypeT4]
    ): ...
    @overload
    def __new__(
        cls,
        types: tuple[_TypeT1, _TypeT2, _TypeT3, _TypeT4, _TypeT5],
    ) -> (
        StarArgUniTuple[_TypeT1 | _TypeT2 | _TypeT3 | _TypeT4 | _TypeT5]
        | StarArgTuple[_TypeT1, _TypeT2, _TypeT3, _TypeT4, _TypeT5]
    ): ...
    @overload
    def __new__(
        cls,
        types: _PyIterable[_TypeT],
    ) -> StarArgUniTuple[_TypeT] | StarArgTuple[Unpack[tuple[_TypeT, ...]]]: ...

class StarArgUniTuple(_StarArgTupleMixin, UniTuple[_TypeT_co], Generic[_TypeT_co]): ...

#
class BaseNamedTuple(BaseTuple): ...

@type_check_only
class _HasFields(Protocol):
    @property
    def __name__(self) -> str: ...
    @property
    def _fields(self) -> tuple[str, ...]: ...

_NamedTupleLike: TypeAlias = type[tuple[Any, ...]] | _HasFields

class NamedUniTuple(_HomogeneousTuple[_TypeT_co], BaseNamedTuple, Generic[_TypeT_co]):
    dtype: _TypeT_co
    count: int
    fields: tuple[str, ...]
    instance_class: _NamedTupleLike

    def __init__(self, dtype: _TypeT_co, count: int, cls: _NamedTupleLike) -> None: ...

    #
    @property
    @override
    def iterator_type(self) -> UniTupleIter[_TypeT_co]: ...
    @property
    @override
    def key(self) -> tuple[_NamedTupleLike, _TypeT_co, int]: ...

class NamedTuple(
    _HeterogeneousTuple[Unpack[_TypeTs]],
    BaseNamedTuple,
    Generic[Unpack[_TypeTs]],
):
    types: tuple[Unpack[_TypeTs]]
    count: int
    fields: tuple[str, ...]
    instance_class: _NamedTupleLike

    @overload
    def __init__(self, types: tuple[Unpack[_TypeTs]], cls: _NamedTupleLike) -> None: ...
    @overload
    def __init__(
        self: NamedTuple[Unpack[tuple[_TypeT, ...]]],
        types: _PyIterable[_TypeT],
        cls: _NamedTupleLike,
    ) -> None: ...

    #
    @property
    @override
    def key(self) -> tuple[_NamedTupleLike, tuple[Unpack[_TypeTs]]]: ...

class List(
    MutableSequence[_TypeT_co],
    InitialValue[_InitT_co],
    Generic[_TypeT_co, _InitT_co],
):
    dtype: _TypeT_co
    reflected: bool

    def __init__(
        self,
        dtype: _TypeT_co,
        reflected: bool = False,
        initial_value: _InitT_co | None = None,
    ) -> None: ...

    #
    @overload
    def copy(
        self,
        dtype: None = None,
        reflected: bool | None = None,
    ) -> List[_TypeT_co, _InitT_co]: ...
    @overload
    def copy(
        self,
        dtype: _TypeT,
        reflected: bool | None = None,
    ) -> List[_TypeT, _InitT_co]: ...

    #
    @property
    @override
    def key(self) -> tuple[_TypeT_co, bool, str]: ...
    @property
    @override
    def iterator_type(self) -> ListIter[_TypeT_co]: ...

    #
    @override
    def __getitem__(self, args: Unused) -> _TypeT_co: ...  # pyrefly:ignore[bad-override]
    def __unliteral__(self) -> List[_TypeT_co, None]: ...

# we can't use `list[_TypeT_co]` here because `list` is invariant
class LiteralList(Literal[_PySequence[_TypeT_co]], ConstSized, Hashable):
    mutable: bool = False

    types: tuple[_TypeT_co, ...]
    count: int
    name: str

    @classmethod
    def from_types(cls, tys: _PyIterable[_TypeT]) -> LiteralList[_TypeT]: ...
    @staticmethod
    def is_types_iterable(types: _PyIterable[object]) -> None: ...

    #
    def __init__(self, literal_value: _PyIterable[_TypeT_co]) -> None: ...
    @override
    def __getitem__(self, i: SupportsIndex) -> _TypeT_co: ...  # pyrefly:ignore[bad-override]
    @override
    def __len__(self) -> int: ...
    def __iter__(self) -> _PyIterator[_TypeT_co]: ...

    #
    @property
    def iterator_type(self) -> ListIter[_TypeT_co]: ...
    def __unliteral__(self) -> Poison[Self]: ...

class ListIter(BaseContainerIterator[_TypeT_co], Generic[_TypeT_co]):
    container_class: ClassVar[type[List]] = ...

class ListPayload(BaseContainerPayload[_ListT_co], Generic[_ListT_co]):
    container_class: ClassVar[type[List]] = ...

class Set(Container[_TypeT_co]):
    mutable: bool = True

    dtype: _TypeT_co
    reflected: bool

    def __init__(self, dtype: _TypeT_co, reflected: bool = False) -> None: ...

    #
    @property
    @override
    def key(self) -> tuple[_TypeT_co, bool]: ...
    @property
    @override
    def iterator_type(self) -> SetIter[_TypeT_co]: ...

    #
    @overload
    def copy(
        self,
        dtype: None = None,
        reflected: bool | None = None,
    ) -> Set[_TypeT_co]: ...
    @overload
    def copy(self, dtype: _TypeT, reflected: bool | None = None) -> Set[_TypeT]: ...

class SetIter(BaseContainerIterator[_TypeT_co], Generic[_TypeT_co]):
    container_class: ClassVar[type[Set]] = ...

class SetPayload(BaseContainerPayload[_SetT_co], Generic[_SetT_co]):
    container_class: ClassVar[type[Set]] = ...

class SetEntry(Type, Generic[_SetT_co]):
    set_type: _SetT_co

    def __init__(self, set_type: _SetT_co) -> None: ...

    #
    @property
    @override
    def key(self) -> _SetT_co: ...

class ListType(IterableType[_TypeT_co], Generic[_TypeT_co]):
    mutable: bool = True

    item_type: _TypeT_co
    dtype: _TypeT_co

    @classmethod
    def refine(cls, itemty: _TypeT) -> ListType[_TypeT]: ...

    #
    def __init__(self, itemty: _TypeT_co) -> None: ...

    #
    @property
    @override
    def key(self) -> _TypeT_co: ...
    @property
    @override
    def iterator_type(self) -> ListTypeIteratorType[_TypeT_co]: ...

class ListTypeIterableType(SimpleIterableType[_TypeT_co], Generic[_TypeT_co]):
    parent: ListType[_TypeT_co]
    yield_type: _TypeT_co

    def __init__(self, parent: ListType[_TypeT_co]) -> None: ...

    #
    @property
    @override
    def iterator_type(self) -> ListTypeIteratorType[_TypeT_co]: ...

class ListTypeIteratorType(SimpleIteratorType[_TypeT_co], Generic[_TypeT_co]):
    parent: ListType[_TypeT_co]
    iterable: ListTypeIterableType[_TypeT_co]

    def __init__(self, iterable: ListTypeIterableType[_TypeT_co]) -> None: ...

class DictType(
    IterableType[_KeyTypeT_co],
    InitialValue[_InitT_co],
    Generic[_KeyTypeT_co, _ValTypeT_co, _InitT_co],
):
    key_type: _KeyTypeT_co
    value_type: _ValTypeT_co
    keyvalue_type: Tuple[_KeyTypeT_co, _ValTypeT_co]

    @classmethod
    def refine(
        cls,
        keyty: _TypeT1,
        valty: _TypeT2,
    ) -> DictType[_TypeT1, _TypeT2, None]: ...

    #
    def __init__(
        self,
        keyty: _KeyTypeT_co,
        valty: _ValTypeT_co,
        initial_value: _InitT_co | None = None,
    ) -> None: ...
    def __unliteral__(self) -> DictType[_KeyTypeT_co, _ValTypeT_co, None]: ...

    #
    @property
    @override
    def iterator_type(self) -> DictIteratorType[_KeyTypeT_co]: ...
    @property
    @override
    def key(self) -> tuple[_KeyTypeT_co, _ValTypeT_co, str]: ...

class LiteralStrKeyDict(
    Literal[_PyMapping[Literal[str], _ValTypeT_co]],
    ConstSized,
    Hashable,
    Generic[_ValTypeT_co],
):
    class FakeNamedTuple(_PySequence[str]):
        __name__: str
        _fields: tuple[str, ...]

        def __init__(self, name: str, keys: _PyIterable[str]) -> None: ...
        @override
        def __len__(self) -> int: ...
        @override
        @overload
        def __getitem__(self, key: SupportsIndex) -> str: ...
        @overload
        def __getitem__(self, key: slice) -> tuple[str, ...]: ...

    value_index: _PyMapping[str, int] | None
    tuple_ty: FakeNamedTuple
    types: tuple[_ValTypeT_co, ...]
    count: int
    fields: tuple[str, ...]
    instance_class: FakeNamedTuple  # = tuple_ty

    def __init__(
        self,
        literal_value: _PyMapping[Literal[str], _ValTypeT_co],
        value_index: _PyMapping[str, int] | None = None,
    ) -> None: ...

    #
    def __unliteral__(self) -> Poison[Self]: ...

    #
    @override
    def __len__(self) -> int: ...
    def __iter__(self) -> _PyIterator[_ValTypeT_co]: ...

    #
    @property
    @override
    def key(self) -> tuple[tuple[str, ...], tuple[_ValTypeT_co, ...], str]: ...

class DictItemsIterableType(
    SimpleIterableType[Tuple[_KeyTypeT_co, _ValTypeT_co]],  # pyrefly:ignore[invalid-variance]
    Generic[_KeyTypeT_co, _ValTypeT_co],
):
    parent: DictType[_KeyTypeT_co, _ValTypeT_co, Any]
    yield_type: Tuple[_KeyTypeT_co, _ValTypeT_co]

    def __init__(self, parent: DictType[_KeyTypeT_co, _ValTypeT_co]) -> None: ...

    #
    @property
    @override
    def iterator_type(self) -> DictIteratorType[Tuple[_KeyTypeT_co, _ValTypeT_co]]: ...

class DictKeysIterableType(SimpleIterableType[_KeyTypeT_co], Generic[_KeyTypeT_co]):
    parent: DictType[_KeyTypeT_co, Any]
    yield_type: _KeyTypeT_co

    def __init__(self, parent: DictType[_KeyTypeT_co, Any]) -> None: ...

    #
    @property
    @override
    def iterator_type(self) -> DictIteratorType[_KeyTypeT_co]: ...

class DictValuesIterableType(SimpleIterableType[_ValTypeT_co], Generic[_ValTypeT_co]):
    parent: DictType[Any, _ValTypeT_co]
    yield_type: _ValTypeT_co

    def __init__(self, parent: DictType[Any, _ValTypeT_co]) -> None: ...

    #
    @property
    @override
    def iterator_type(self) -> DictIteratorType[_ValTypeT_co]: ...

class DictIteratorType(SimpleIteratorType[_TypeT_co]):
    parent: DictType
    iterable: SimpleIterableType[_TypeT_co]

    def __init__(self, iterable: SimpleIterableType[_TypeT_co]) -> None: ...

class SetType(IterableType[_KeyTypeT_co], InitialValue[_InitT_co]):
    key_type: _KeyTypeT_co

    @classmethod
    def refine(cls, keyty: _TypeT) -> SetType[_TypeT, None]: ...

    #
    def __init__(
        self,
        keyty: _KeyTypeT_co,
        initial_value: _InitT_co | None = None,
    ) -> None: ...
    def __unliteral__(self) -> SetType[_KeyTypeT_co, None]: ...

    #
    @property
    @override
    def iterator_type(self) -> SetIteratorType[_KeyTypeT_co]: ...
    @property
    @override
    def key(self) -> tuple[_KeyTypeT_co, str]: ...

class SetIterableType(SimpleIterableType[_KeyTypeT_co]):
    parent: SetType[_KeyTypeT_co]
    yield_type: _KeyTypeT_co

    def __init__(self, parent: SetType[_KeyTypeT_co]) -> None: ...

    #
    @property
    @override
    def iterator_type(self) -> SetIteratorType[_KeyTypeT_co]: ...

class SetIteratorType(SimpleIteratorType[_KeyTypeT_co]):
    parent: SetType[_KeyTypeT_co]
    iterable: SetIterableType[_KeyTypeT_co]

    def __init__(self, iterable: SetIterableType[_KeyTypeT_co]) -> None: ...

class StructRef(Type, Generic[_TypeT_co]):
    _fields: tuple[tuple[str, _TypeT_co], ...]
    _typename: str

    def __init__(self, fields: _PyIterable[tuple[str, _TypeT_co]]) -> None: ...

    #
    def preprocess_fields(
        self,
        fields: tuple[tuple[str, _TypeT], ...],
    ) -> tuple[tuple[str, _TypeT], ...]: ...

    #
    @property
    def field_dict(self) -> MappingProxyType[str, _TypeT_co]: ...

    #
    def get_data_type(self) -> StructRefPayload[_TypeT_co]: ...

class StructRefPayload(Type, Generic[_TypeT]):
    mutable: bool = True

    _fields: tuple[tuple[str, _TypeT], ...]
    _typename: str

    def __init__(
        self,
        typename: str,
        fields: _PyIterable[tuple[str, _TypeT]],
    ) -> None: ...

    #
    @property
    def field_dict(self) -> MappingProxyType[str, _TypeT]: ...

#
def is_homogeneous(*tys: Type) -> bool: ...
