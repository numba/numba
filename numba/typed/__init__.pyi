from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
)
from typing import Any, Generic, SupportsIndex, TypeAlias, overload

from _typeshed import Incomplete, SupportsRichComparison
from typing_extensions import Self, TypeVar, override

from numba.core import types

###

# `numba.core.runtime.nrt.MemInfo` isn't usable as type (`nrt` has no stubs)
_MemInfo: TypeAlias = Incomplete

_T = TypeVar("_T")
_KT = TypeVar("_KT", default=Any)
_VT = TypeVar("_VT", default=Any)

###

class Dict(MutableMapping[_KT, _VT], Generic[_KT, _VT]):
    def __new__(
        cls,
        # workaround for __new__/__init__ mismatch: also accept iterable for `dcttype`
        dcttype: types.DictType | Iterable[tuple[_KT, _VT]] | None = None,
        meminfo: _MemInfo | None = None,
        n_keys: int = 0,
    ) -> Self: ...
    def __init__(
        self,
        arg0: Iterable[tuple[_KT, _VT]] = ...,
        /,
        *,
        dcttype: types.DictType | None = None,
        meminfo: _MemInfo | None = None,
        n_keys: int = 0,
    ) -> None: ...

    # override abstract `MutableMapping` methods so `Dict` isn't implicitly abstract
    @override
    def __iter__(self) -> Iterator[_KT]: ...
    @override
    def __len__(self) -> int: ...
    @override
    def __getitem__(self, key: _KT, /) -> _VT: ...
    @override
    def __setitem__(self, key: _KT, value: _VT, /) -> None: ...
    @override
    def __delitem__(self, key: _KT, /) -> None: ...

    #
    @classmethod
    def empty(
        cls,
        key_type: types.Type,
        value_type: types.Type,
        n_keys: int = 0,
    ) -> Self: ...

    #
    @override
    @overload
    def get(self, key: _KT, default: None = None) -> _VT | None: ...
    @overload
    def get(self, key: _KT, default: _VT) -> _VT: ...
    @overload
    def get(self, key: _KT, default: _T) -> _VT | _T: ...

    #
    @override
    @overload
    def setdefault(
        self: Dict[_KT, _T | None],
        key: _KT,
        default: None = None,
    ) -> _T | None: ...
    @overload
    def setdefault(self, key: _KT, default: _VT) -> _VT: ...

    #
    def copy(self) -> Self: ...

class List(MutableSequence[_VT], Generic[_VT]):
    def __new__(
        cls,
        iterable: Iterable[_VT] = ...,
        /,
        *,
        lsttype: types.ListType | None = None,
        meminfo: _MemInfo | None = None,
        allocated: int = 0,
    ) -> Self: ...
    def __init__(
        self,
        iterable: Iterable[_VT] = ...,
        /,
        *,
        lsttype: types.ListType | None = None,
        meminfo: _MemInfo | None = None,
        allocated: int = 0,
    ) -> None: ...

    # override abstract `MutableSequence` methods so `List` isn't implicitly abstract
    @override
    def __len__(self) -> int: ...
    @override
    @overload
    def __getitem__(self, index: int, /) -> _VT: ...
    @overload
    def __getitem__(self, index: slice[int | None], /) -> Self: ...
    @override
    @overload
    def __setitem__(self, index: int, value: _VT, /) -> None: ...
    @overload
    def __setitem__(
        self, index: slice[int | None], value: Iterable[_VT], /
    ) -> None: ...
    @override
    @overload
    def __delitem__(self, index: int, /) -> None: ...
    @overload
    def __delitem__(self, index: slice[int | None], /) -> None: ...

    #
    @classmethod
    def empty_list(cls, item_type: types.Type, allocated: int = 0) -> Self: ...

    #
    def __lt__(self, other: List[Any]) -> bool: ...
    def __le__(self, other: List[Any]) -> bool: ...
    def __gt__(self, other: List[Any]) -> bool: ...
    def __ge__(self, other: List[Any]) -> bool: ...

    #
    @override
    def append(self, item: _VT) -> None: ...
    @override
    def index(
        self,
        item: object,
        start: int | None = None,
        stop: int | None = None,
    ) -> int: ...
    @override
    def insert(self, i: int, item: _VT) -> None: ...
    @override
    def count(self, item: object) -> int: ...
    @override
    def pop(self, i: SupportsIndex = -1) -> _VT: ...
    @override
    def extend(self, iterable: Sequence[_VT]) -> None: ...  # pyrefly:ignore[bad-override]
    @override
    def remove(self, item: _VT) -> None: ...

    #
    def sort(
        self,
        key: Callable[[_VT], SupportsRichComparison] | None = None,
        reverse: bool = False,
    ) -> None: ...
    def copy(self) -> Self: ...

class Set(MutableSet[_VT], Generic[_VT]):
    def __new__(
        cls,
        settype: types.SetType | None = None,
        meminfo: _MemInfo | None = None,
    ) -> Self: ...
    def __init__(
        self,
        *,
        settype: types.SetType | None = None,
        meminfo: _MemInfo | None = None,
    ) -> None: ...

    # override abstract `MutableSet` methods so `Set` isn't implicitly abstract
    @override
    def __len__(self) -> int: ...
    @override
    def __iter__(self) -> Iterator[_VT]: ...
    @override
    def __contains__(self, x: object, /) -> bool: ...

    #
    @classmethod
    def empty(cls, key_type: types.Type) -> Self: ...

    #
    @override
    def add(self, key: _VT) -> None: ...
    @override
    def discard(self, key: _VT) -> None: ...

    #
    def copy(self) -> Self: ...
