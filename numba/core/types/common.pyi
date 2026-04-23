from typing import ClassVar, Literal, TypeAlias

from typing_extensions import Generic, Self, TypeVar, override

from .abstract import ArrayCompatible, Dummy, IterableType, IteratorType, Type
from .iterators import ArrayIterator

_TypeT_co = TypeVar("_TypeT_co", bound=Type, default=Type, covariant=True)

_Layout: TypeAlias = Literal["C", "F", "CS", "FS", "A"]

###

class Opaque(Dummy): ...

class SimpleIterableType(IterableType[_TypeT_co], Generic[_TypeT_co]):
    @override
    def __init__(self, name: str, iterator_type: IteratorType[_TypeT_co]): ...
    @property
    @override
    def iterator_type(self) -> IteratorType[_TypeT_co]: ...

class SimpleIteratorType(IteratorType[_TypeT_co], Generic[_TypeT_co]):
    @override
    def __init__(self, name: str, yield_type: _TypeT_co): ...
    @property
    @override
    def yield_type(self) -> _TypeT_co: ...

class Buffer(IterableType, ArrayCompatible):
    LAYOUTS: ClassVar[frozenset[_Layout]] = ...

    slice_is_copy: ClassVar[bool] = False
    aligned: ClassVar[bool] = True

    mutable: bool = True

    @override
    def __init__(
        self,
        dtype: Type,
        ndim: int,
        layout: _Layout,
        readonly: bool = False,
        name: str | None = None,
    ) -> None: ...
    @property  # TODO: Use generic `ArrayIterator` once we have `iterators.pyi`
    @override
    def iterator_type(self) -> ArrayIterator: ...
    @property
    @override
    def as_array(self) -> Self: ...
    @property
    @override
    def key(self) -> tuple[Type, int, _Layout, bool]: ...

    #
    @property
    def is_c_contig(self) -> bool: ...
    @property
    def is_f_contig(self) -> bool: ...
    @property
    def is_contig(self) -> bool: ...

    #
    def copy(
        self,
        dtype: Type | None = None,
        ndim: int | None = None,
        layout: _Layout | None = None,
    ) -> Self: ...
