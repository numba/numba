from types import TracebackType
from typing import Any, ClassVar, Final, Generic, Protocol, type_check_only

from _typeshed import Unused
from typing_extensions import Never, Self, TypeVar

from numba.core import dispatcher, ir, types

###

_ExtraT_contra = TypeVar("_ExtraT_contra", default=Never, contravariant=True)
_DispatcherT_co = TypeVar("_DispatcherT_co", default=Any, covariant=True)

@type_check_only
class _DispatcherFactory(Protocol):
    def __call__(
        self,
        func_ir: ir.FunctionIR,
        /,
        objectmode: bool = ...,
        *,
        output_types: types.Type = ...,
    ) -> dispatcher._DispatcherBase: ...

###

class WithContext(Generic[_ExtraT_contra, _DispatcherT_co]):
    is_callable: ClassVar[bool] = ...

    def __enter__(self) -> None: ...
    def __exit__(
        self,
        typ: type[BaseException] | None,
        val: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    # abstract-ish
    def mutate_with_body(
        self,
        func_ir: ir.FunctionIR,
        blocks: dict[int, ir.Block],
        blk_start: int,
        blk_end: int,
        body_blocks: list[int],
        dispatcher_factory: _DispatcherFactory,
        extra: _ExtraT_contra,
    ) -> _DispatcherT_co: ...

class _ByPassContextType(WithContext[None, None]): ...
class _CallContextType(WithContext[None, dispatcher.LiftedWith]): ...

class _ObjModeContextType(
    WithContext[
        dict[str, Any] | None,
        dispatcher.ObjModeLiftedWith,
    ],
):
    def __call__(self, *args: Unused, **kwargs: Unused) -> Self: ...

class _ParallelChunksize(WithContext[dict[str, Any], None]):
    chunksize: int
    orig_chunksize: int

    def __call__(self, chunksize: int, /) -> Self: ...

# undocumented
def typeof_contextmanager(val: WithContext, c: Unused) -> types.ContextManager: ...

bypass_context: Final[_ByPassContextType] = ...  # undocumented
call_context: Final[_CallContextType] = ...  # undocumented

objmode_context: Final[_ObjModeContextType] = ...  # = numba.objmode
parallel_chunksize: Final[_ParallelChunksize] = ...  # = numba.parallel_chunksize
