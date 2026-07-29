import functools
from collections.abc import Iterable as _PyIterable
from typing import Any, Generic, NamedTuple, TypeAlias, overload
from typing import Literal as _PyLiteral

import numpy as np
from _typeshed import Unused
from typing_extensions import NotRequired, TypedDict, TypeVar, override

from .abstract import DTypeSpec, IteratorType, MutableSequence, Number, Type, _SpecSlice
from .common import Buffer, Opaque, SimpleIteratorType, _Layout
from .containers import Tuple, UniTuple

###

class _RecordFieldDict(TypedDict):
    type: Type
    offset: int
    alignment: NotRequired[int | None]
    title: NotRequired[str | None]

_ToFields: TypeAlias = _PyIterable[tuple[str, _RecordFieldDict]]
_IndexKind: TypeAlias = _PyLiteral["flat", "indexed", "0d", "scalar"]

_TypeT = TypeVar("_TypeT", bound=Type)
_TypeT_co = TypeVar("_TypeT_co", bound=Type, default=Type, covariant=True)
_ArrayTypeT_co = TypeVar("_ArrayTypeT_co", bound=Array, default=Array, covariant=True)
_ShapeT_co = TypeVar(
    "_ShapeT_co",
    bound=tuple[int, ...],
    default=tuple[Any, ...],
    covariant=True,
)

###
# ruff: noqa: PLR2044

class CharSeq(Type):
    mutable: bool = True

    count: int

    def __init__(self, count: int) -> None: ...

    #
    @property
    @override
    def key(self) -> int: ...

class UnicodeCharSeq(Type):
    mutable: bool = True

    count: int

    def __init__(self, count: int) -> None: ...

    #
    @property
    @override
    def key(self) -> int: ...

class _RecordField(NamedTuple):
    type: Type
    offset: int
    alignment: int | None
    title: str | None

class Record(Type):
    mutable: bool = True

    fields: dict[str, _RecordField]
    size: int
    aligned: bool
    bitwidth: int

    @classmethod
    def make_c_struct(
        cls,
        name_types: _PyIterable[tuple[str, Number | NestedArray]],
    ) -> Record: ...

    #
    def __init__(self, fields: _ToFields, size: int, aligned: bool) -> None: ...

    #
    @property
    @override
    def key(self) -> str: ...

    #
    def __len__(self) -> int: ...
    def offset(self, key: str) -> int: ...
    def typeof(self, key: str) -> Type: ...
    def alignof(self, key: str) -> int | None: ...
    def has_titles(self) -> bool: ...
    def is_title(self, key: str) -> bool: ...

    #
    @property
    def members(self) -> list[tuple[str, Type]]: ...
    @property
    def dtype(self) -> np.dtype[np.void]: ...

class DType(DTypeSpec, Opaque, Generic[_TypeT_co]):
    _dtype: _TypeT_co

    def __init__(self, dtype: _TypeT_co) -> None: ...

    #
    @override
    def __getitem__(
        self,
        arg: _SpecSlice | tuple[_SpecSlice, ...] | list[_SpecSlice],
    ) -> Array[_TypeT_co]: ...

    #
    @property
    @override
    def dtype(self) -> _TypeT_co: ...
    @property
    @override
    def key(self) -> _TypeT_co: ...

class NumpyFlatType(
    SimpleIteratorType[_TypeT_co],
    MutableSequence[_TypeT_co],
    Generic[_TypeT_co],
):
    array_type: Array[_TypeT_co]
    dtype: _TypeT_co

    def __init__(self, arrty: Array[_TypeT_co]) -> None: ...

    #
    @property
    @override
    def key(self) -> Array[_TypeT_co]: ...

class NumpyNdEnumerateType(SimpleIteratorType[Tuple], Generic[_ArrayTypeT_co]):
    array_type: _ArrayTypeT_co

    def __init__(self, arrty: _ArrayTypeT_co) -> None: ...

    #
    @property
    @override
    def key(self) -> _ArrayTypeT_co: ...

class NumpyNdIterType(IteratorType):
    arrays: tuple[Array | Number, ...]
    layout: _PyLiteral["F", "C"]
    dtypes: tuple[Type, ...]
    ndim: int

    def __init__(self, arrays: _PyIterable[Array | Number]) -> None: ...

    #
    @property
    @override
    def key(self) -> tuple[Array | Number, ...]: ...
    @property
    @override
    def yield_type(self) -> Type: ...
    @property
    def views(self) -> list[Array]: ...

    #
    @functools.cached_property
    def indexers(self) -> list[tuple[_IndexKind, int, int, list[int]]]: ...
    @functools.cached_property
    def need_shaped_indexing(self) -> bool: ...

class NumpyNdIndexType(SimpleIteratorType[UniTuple]):
    ndim: int

    def __init__(self, ndim: int) -> None: ...

    #
    @property
    @override
    def key(self) -> int: ...

class Array(Buffer, Generic[_TypeT_co]):
    @overload
    def __init__(
        self: Array[_TypeT],
        dtype: NestedArray[_TypeT, Any],
        ndim: int,
        layout: _Layout,
        readonly: bool = False,
        name: str | None = None,
        aligned: bool = True,
    ) -> None: ...
    @overload
    def __init__(
        self,
        dtype: _TypeT_co,
        ndim: int,
        layout: _Layout,
        readonly: bool = False,
        name: str | None = None,
        aligned: bool = True,
    ) -> None: ...

    #
    @override
    @overload
    def copy(  # pyrefly:ignore[bad-override]
        self,
        dtype: None = None,
        ndim: int | None = None,
        layout: _Layout | None = None,
        readonly: bool | None = None,
    ) -> Array[_TypeT_co]: ...
    @overload
    def copy(
        self,
        dtype: _TypeT,
        ndim: int | None = None,
        layout: _Layout | None = None,
        readonly: bool | None = None,
    ) -> Array[_TypeT]: ...

    #
    @functools.cached_property
    @override
    def dtype(self) -> _TypeT_co: ...
    @property
    @override
    def key(self) -> tuple[_TypeT_co, int, _Layout, bool, bool]: ...  # pyrefly:ignore[bad-override]
    @property
    def box_type(self) -> type[np.ndarray[tuple[Any, ...], np.dtype[Any]]]: ...

class ArrayCTypes(Type, Generic[_TypeT_co]):
    dtype: _TypeT_co
    ndim: int

    def __init__(self, arytype: Array[_TypeT_co]) -> None: ...

    #
    @property
    @override
    def key(self) -> tuple[_TypeT_co, int]: ...

class ArrayFlags(Type, Generic[_ArrayTypeT_co]):
    array_type: _ArrayTypeT_co

    def __init__(self, arytype: _ArrayTypeT_co) -> None: ...

    #
    @property
    @override
    def key(self) -> _ArrayTypeT_co: ...

class NestedArray(Array[_TypeT_co], Generic[_TypeT_co, _ShapeT_co]):
    _shape: _ShapeT_co

    @overload
    def __init__(
        self: NestedArray[_TypeT],
        dtype: NestedArray[_TypeT, Any],
        shape: tuple[int, ...],
    ) -> None: ...
    @overload
    def __init__(self, dtype: _TypeT_co, shape: _ShapeT_co) -> None: ...

    #
    @property
    def shape(self) -> _ShapeT_co: ...
    @property
    def nitems(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def strides(self) -> tuple[int, ...]: ...
    @property
    @override
    def key(self) -> tuple[_TypeT_co, _ShapeT_co]: ...  # pyrefly:ignore[bad-override]

class NumPyRandomBitGeneratorType(Type):
    def __init__(self, name: Unused, /) -> None: ...

class NumPyRandomGeneratorType(Type):
    def __init__(self, name: Unused, /) -> None: ...

class PolynomialType(Type):
    coef: Array
    domain: Array | None
    window: Array | None
    n_args: int

    def __init__(
        self,
        coef: Array,
        domain: Array | None = None,
        window: Array | None = None,
        n_args: int = 1,
    ) -> None: ...
