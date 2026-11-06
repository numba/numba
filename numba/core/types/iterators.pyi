from collections.abc import Callable as _PyCallable
from collections.abc import Iterable as _PyIterable
from typing import Any, Generic

from typing_extensions import TypeVar, override

from .abstract import IterableType, IteratorType, Type
from .common import Buffer, SimpleIterableType, SimpleIteratorType
from .containers import BaseAnonymousTuple
from .scalars import Number

###

_TypeT_co = TypeVar("_TypeT_co", bound=Type, default=Type, covariant=True)
_NumberT_co = TypeVar("_NumberT_co", bound=Number, default=Number, covariant=True)
_BufferT_co = TypeVar("_BufferT_co", bound=Buffer, default=Buffer, covariant=True)

###

class RangeType(SimpleIterableType[_NumberT_co], Generic[_NumberT_co]):
    dtype: _NumberT_co

    def __init__(self, dtype: _NumberT_co) -> None: ...
    
    #
    @property
    @override
    def iterator_type(self) -> RangeIteratorType[_NumberT_co]: ...

class RangeIteratorType(SimpleIteratorType[_NumberT_co], Generic[_NumberT_co]):
    def __init__(self, dtype: _NumberT_co) -> None: ...

class Generator(SimpleIteratorType[_TypeT_co], Generic[_TypeT_co]):
    gen_func: _PyCallable[..., Any]
    arg_types: tuple[Type, ...]
    state_types: tuple[Type, ...]
    has_finalizer: bool

    def __init__(
        self,
        gen_func: _PyCallable[..., Any],
        yield_type: _TypeT_co,
        arg_types: _PyIterable[Type],
        state_types: _PyIterable[Type],
        has_finalizer: bool,
    ) -> None: ...

    #
    @property
    @override
    def key(self) -> tuple[
        _PyCallable[..., Any],
        tuple[Type, ...],
        _TypeT_co,
        bool,
        tuple[Type, ...],
    ]: ...  # fmt: skip

class EnumerateType(SimpleIteratorType[BaseAnonymousTuple], Generic[_TypeT_co]):
    source_type: IteratorType[_TypeT_co]

    def __init__(self, iterable_type: IterableType[_TypeT_co]) -> None: ...

    #
    @property
    @override
    def key(self) -> IteratorType[_TypeT_co]: ...

class ZipType(SimpleIteratorType[BaseAnonymousTuple], Generic[_TypeT_co]):
    source_types: tuple[IteratorType[_TypeT_co], ...]

    def __init__(
        self,
        iterable_types: _PyIterable[IterableType[_TypeT_co]],
    ) -> None: ...

    #
    @property
    @override
    def key(self) -> tuple[IteratorType[_TypeT_co], ...]: ...

class ArrayIterator(SimpleIteratorType[Type], Generic[_BufferT_co]):
    array_type: _BufferT_co

    def __init__(self, array_type: _BufferT_co) -> None: ...
