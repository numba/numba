from collections.abc import Callable as _PyCallable
from collections.abc import Mapping as _PyMapping
from types import ModuleType as _PyModule
from typing import (
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeAlias,
    overload,
    type_check_only,
)
from typing import Literal as _PyLiteral

from _typeshed import Unused
from typing_extensions import Self, TypedDict, TypeVar, override

from numba.core.dispatcher import Dispatcher
from numba.core.typing import context, templates
from numba.core.withcontexts import WithContext

from .abstract import Callable, Dummy, Hashable, IterableType, Literal, Type
from .common import Opaque, SimpleIteratorType
from .scalars import BooleanLiteral, IntegerLiteral

###

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", default=Any, covariant=True)

_StartT = TypeVar("_StartT")
_StopT = TypeVar("_StopT")
_StepT = TypeVar("_StepT")

_StartT_co = TypeVar("_StartT_co", covariant=True, default=Any)
_StopT_co = TypeVar("_StopT_co", covariant=True, default=_StartT_co)
_StepT_co = TypeVar("_StepT_co", covariant=True, default=_StartT_co | _StopT_co)

_DType: TypeAlias = Type | type[Type]

_TypeT = TypeVar("_TypeT", bound=Type, default=Any)
_TypeT_co = TypeVar("_TypeT_co", bound=Type, default=Type, covariant=True)
_DTypeT_co = TypeVar("_DTypeT_co", bound=_DType, default=Any, covariant=True)
_CMT_co = TypeVar("_CMT_co", bound=WithContext, default=WithContext, covariant=True)

_PyModT_co = TypeVar("_PyModT_co", bound=_PyModule | type, default=Any, covariant=True)
_PyExcT_co = TypeVar(
    "_PyExcT_co",
    bound=BaseException,
    default=BaseException,
    covariant=True,
)

@type_check_only
class _CanUnliteral(Protocol[_T_co]):
    def __unliteral__(self, /) -> _T_co: ...

@type_check_only
class _HasLiteralType(Protocol[_T_co]):
    @property
    def literal_type(self, /) -> _T_co: ...

@type_check_only
class _JitProp(TypedDict, total=False):
    get: Dispatcher
    set: Dispatcher

###
# ruff: noqa: PLR2044

class PyObject(Dummy): ...
class Phantom(Dummy): ...
class Undefined(Dummy): ...
class UndefVar(Dummy): ...
class RawPointer(Opaque): ...
class StringLiteral(Literal[str], Dummy): ...

@overload
def unliteral(lit_type: _CanUnliteral[_T]) -> _T: ...
@overload
def unliteral(lit_type: _HasLiteralType[_T]) -> _T: ...
@overload
def unliteral(lit_type: _T) -> _T: ...

#
@overload
def literal(value: str) -> StringLiteral: ...
@overload
def literal(
    value: slice[_StartT, _StopT, _StepT],
) -> SliceLiteral[_StartT, _StopT, _StepT]: ...
@overload
def literal(value: bool) -> BooleanLiteral: ...
@overload
def literal(value: int) -> IntegerLiteral: ...

#
@overload
def maybe_literal(value: str) -> StringLiteral: ...
@overload
def maybe_literal(
    value: slice[_StartT, _StopT, _StepT],
) -> SliceLiteral[_StartT, _StopT, _StepT]: ...
@overload
def maybe_literal(value: bool) -> BooleanLiteral: ...
@overload
def maybe_literal(value: int) -> IntegerLiteral: ...
@overload  # `object` overlaps with the above; `Any | ` avoids incompatible return types
def maybe_literal(value: object) -> Any | None: ...

class Omitted(Opaque, Generic[_T_co]):
    _value: _T_co
    _value_key: _T_co | int  # `int` iff not hashable

    def __init__(self, value: _T_co) -> None: ...

    #
    @property
    @override
    def key(self) -> tuple[type[_T_co], _T_co | int]: ...
    @property
    def value(self) -> _T_co: ...

class VarArg(Type, Generic[_DTypeT_co]):
    dtype: _DTypeT_co

    def __init__(self, dtype: _DTypeT_co) -> None: ...

    #
    @property
    @override
    def key(self) -> _DTypeT_co: ...

class Module(Dummy, Generic[_PyModT_co]):
    pymod: _PyModT_co

    def __init__(self, pymod: _PyModT_co) -> None: ...

    #
    @property
    @override
    def key(self) -> _PyModT_co: ...

class MemInfoPointer(Type, Generic[_TypeT]):
    mutable: bool = True

    dtype: _TypeT

    def __init__(self, dtype: _TypeT) -> None: ...

    #
    @property
    @override
    def key(self) -> _TypeT: ...

class CPointer(Type, Generic[_TypeT]):
    mutable: bool = True

    dtype: _TypeT
    addrspace: int | None

    def __init__(self, dtype: _TypeT, addrspace: int | None = None) -> None: ...

    #
    @property
    @override
    def key(self) -> tuple[_TypeT, int | None]: ...

class EphemeralPointer(CPointer[_TypeT], Generic[_TypeT]): ...

class EphemeralArray(Type, Generic[_TypeT_co]):
    dtype: _TypeT_co
    count: int

    def __init__(self, dtype: _TypeT_co, count: int) -> None: ...

    #
    @property
    @override
    def key(self) -> tuple[_TypeT_co, int]: ...

class Object(Type):
    mutable: bool = True

    cls: type

    def __init__(self, clsobj: type) -> None: ...

    #
    @property
    @override
    def key(self) -> type: ...

class Optional(Type, Generic[_TypeT_co]):
    type: _TypeT_co

    @overload
    def __init__(self, typ: _CanUnliteral[_TypeT_co]) -> None: ...
    @overload
    def __init__(self, typ: _HasLiteralType[_TypeT_co]) -> None: ...
    @overload
    def __init__(self, typ: _TypeT_co) -> None: ...

    #
    @property
    @override
    def key(self) -> _TypeT_co: ...

class NoneType(Opaque): ...
class EllipsisType(Opaque): ...

class ExceptionClass(Callable, Phantom, Generic[_PyExcT_co]):
    exc_class: type[_PyExcT_co]

    def __init__(self, exc_class: type[_PyExcT_co]) -> None: ...

    #
    @override
    def get_call_type(
        self,
        context: Unused,
        args: Unused,
        kws: Unused,
    ) -> templates.Signature: ...
    @override
    def get_call_signatures(
        self,
    ) -> tuple[list[templates.Signature], _PyLiteral[False]]: ...
    @override
    def get_impl_key(self, sig: Unused) -> type[Self]: ...

    #
    @property
    @override
    def key(self) -> type[_PyExcT_co]: ...

class ExceptionInstance(Phantom, Generic[_PyExcT_co]):
    exc_class: type[_PyExcT_co]

    def __init__(self, exc_class: type[_PyExcT_co]) -> None: ...

    #
    @property
    @override
    def key(self) -> type[_PyExcT_co]: ...

class SliceType(Type):
    members: _PyLiteral[2, 3]
    has_step: bool

    def __init__(self, name: str, members: _PyLiteral[2, 3]) -> None: ...

    #
    @property
    @override
    def key(self) -> _PyLiteral[2, 3]: ...

class SliceLiteral(
    Literal[slice[_StartT_co, _StopT_co, _StepT_co]],
    SliceType,
    Generic[_StartT_co, _StopT_co, _StepT_co],
):
    def __init__(self, value: slice[_StartT_co, _StopT_co, _StepT_co]) -> None: ...

    #
    @property
    @override
    def key(self) -> tuple[_StartT_co, _StopT_co, _StepT_co]: ...  # pyrefly:ignore[bad-override]

class ClassInstanceType(Type, Generic[_T_co]):
    mutable: bool = True
    name_prefix: str = "instance"

    class_type: ClassType[_T_co]

    def __init__(self, class_type: ClassType[_T_co]) -> None: ...

    #
    def get_data_type(self) -> ClassDataType[_T_co]: ...
    def get_reference_type(self) -> Self: ...

    #
    @property
    @override
    def key(self) -> str: ...
    @property
    def classname(self) -> str: ...
    @property
    def jit_props(self) -> _PyMapping[str, _JitProp]: ...
    @property
    def jit_static_methods(self) -> _PyMapping[str, Dispatcher]: ...
    @property
    def jit_methods(self) -> _PyMapping[str, Dispatcher]: ...
    @property
    def struct(self) -> _PyMapping[str, Type]: ...
    @property
    def methods(self) -> dict[str, _PyCallable[..., Any]]: ...
    @property
    def static_methods(self) -> dict[str, _PyCallable[..., Any]]: ...

class ClassType(Callable, Opaque, Generic[_T_co]):
    mutable: bool = True
    name_prefix: str = "jitclass"
    instance_type_class: ClassVar[type[ClassInstanceType]] = ...

    class_name: str
    class_doc: str | None
    _ctor_template_class: type[templates.AbstractTemplate]
    jit_methods: _PyMapping[str, Dispatcher]
    jit_props: _PyMapping[str, _JitProp]
    jit_static_methods: _PyMapping[str, Dispatcher]
    struct: _PyMapping[str, Type]

    def __init__(
        self,
        class_def: type[_T_co],
        ctor_template_cls: type[templates.AbstractTemplate],
        struct: _PyMapping[str, Type],
        jit_methods: _PyMapping[str, Dispatcher],
        jit_props: _PyMapping[str, _JitProp],
        jit_static_methods: _PyMapping[str, Dispatcher],
    ) -> None: ...

    #
    @override
    def get_call_type(
        self,
        context: context.Context,
        args: tuple[Type, ...],
        kws: _PyMapping[str, Type],
    ) -> templates.Signature: ...
    @override
    def get_call_signatures(self) -> tuple[tuple[()], _PyLiteral[True]]: ...
    @override
    def get_impl_key(self, sig: Unused) -> type[Self]: ...

    #
    @property
    def methods(self) -> dict[str, _PyCallable[..., Any]]: ...
    @property
    def static_methods(self) -> dict[str, _PyCallable[..., Any]]: ...
    @property
    def instance_type(self) -> ClassInstanceType[_T_co]: ...
    @property
    def ctor_template(self) -> templates.AbstractTemplate: ...

class ClassDataType(Type, Generic[_T_co]):
    class_type: ClassInstanceType[_T_co]

    def __init__(self, classtyp: ClassInstanceType[_T_co]) -> None: ...

class DeferredType(Type, Generic[_TypeT]):
    _define: _TypeT | None

    def __init__(self) -> None: ...
    def get(self) -> _TypeT: ...
    def define(self, typ: _TypeT) -> None: ...

class ContextManager(Callable, Phantom, Generic[_CMT_co]):
    cm: _CMT_co

    def __init__(self, cm: _CMT_co) -> None: ...

    #
    @override
    def get_call_type(
        self,
        context: Unused,
        args: tuple[Type, ...],
        kws: _PyMapping[str, Type],
    ) -> templates.Signature: ...
    @override
    def get_call_signatures(self) -> tuple[tuple[()], _PyLiteral[False]]: ...
    @override
    def get_impl_key(self, sig: Unused) -> type[Self]: ...

class UnicodeType(IterableType[UnicodeType], Hashable):
    def __init__(self, name: str) -> None: ...

    #
    @property
    @override
    def iterator_type(self) -> UnicodeIteratorType: ...

class UnicodeIteratorType(SimpleIteratorType[UnicodeType]):
    data: UnicodeType

    def __init__(self, dtype: UnicodeType) -> None: ...

    #
    @property
    @override
    def yield_type(self) -> UnicodeType: ...
