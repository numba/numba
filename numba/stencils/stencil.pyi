from collections.abc import Callable, Container, Mapping, Sequence
from typing import (
    Any,
    ClassVar,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
    overload,
    type_check_only,
)

import numpy as np
import numpy.typing as npt
from llvmlite import ir as lir

# direct import because pyrefly can't resolve it through lir
from llvmlite.ir.builder import Builder
from typing_extensions import Generic, TypeVar, TypeVarTuple, Unpack

from numba.core import types
from numba.core.base import BaseContext
from numba.core.compiler import CompileResult
from numba.core.ir import Block, Expr, FunctionIR
from numba.core.typing.templates import Signature

_RT = TypeVar("_RT")
_Ts0 = TypeVarTuple("_Ts0")
_Ts = TypeVarTuple("_Ts", default=Unpack[tuple[Any, Unpack[tuple[Any, ...]]]])
_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_ScalarT_co = TypeVar("_ScalarT_co", bound=np.generic, covariant=True, default=Any)
_StencilFuncT_co = TypeVar(
    "_StencilFuncT_co",
    bound=StencilFunc,
    covariant=True,
    default=StencilFunc,
)
_ArrayT = TypeVar("_ArrayT", bound=npt.NDArray[Any])

_Mode: TypeAlias = Literal["constant"]

@type_check_only
class _StencilOptions(TypedDict, total=False):
    cval: complex | np.generic
    standard_indexing: Container[str]
    neighborhood: tuple[tuple[int, int], ...]

# the invariant return type is used to avoid overlapping overloads
@type_check_only
class _InvariantPosonlyCallable(Protocol[_RT, Unpack[_Ts0]]):  # pyrefly: ignore[variance-mismatch]
    def __call__(self, /, *args: Unpack[_Ts0]) -> _RT: ...

@type_check_only
class _StencilWrapper(Protocol):
    @overload
    def __call__(
        self, func: Callable[[Unpack[_Ts0]], _ScalarT]
    ) -> StencilFunc[_ScalarT, Unpack[_Ts0]]: ...
    @overload  # the mypy error is a false positive due to the invariant return type
    def __call__(  # type: ignore[overload-overlap]
        self, func: _InvariantPosonlyCallable[bool, Unpack[_Ts0]]
    ) -> StencilFunc[np.bool_, Unpack[_Ts0]]: ...
    @overload
    def __call__(
        self, func: _InvariantPosonlyCallable[int, Unpack[_Ts0]]
    ) -> StencilFunc[np.int_, Unpack[_Ts0]]: ...
    @overload
    def __call__(
        self, func: _InvariantPosonlyCallable[float, Unpack[_Ts0]]
    ) -> StencilFunc[np.float64, Unpack[_Ts0]]: ...
    @overload
    def __call__(
        self, func: _InvariantPosonlyCallable[complex, Unpack[_Ts0]]
    ) -> StencilFunc[np.complex128, Unpack[_Ts0]]: ...
    @overload
    def __call__(
        self, func: Callable[[Unpack[_Ts0]], Any]
    ) -> StencilFunc[Any, Unpack[_Ts0]]: ...

###

# undocumented
class StencilFuncLowerer(Generic[_StencilFuncT_co]):
    stencilFunc: _StencilFuncT_co

    def __init__(self, sf: _StencilFuncT_co) -> None: ...
    def __call__(
        self,
        context: BaseContext,
        builder: Builder,
        sig: Signature,
        args: Sequence[lir.Value],
    ) -> lir.Value: ...

# undocumented
def stencil_dummy_lower(
    context: BaseContext,
    builder: Builder,
    sig: Signature,
    args: Sequence[lir.Value],
) -> lir.Constant: ...

# undocumented
def raise_if_incompatible_array_sizes(
    a: npt.NDArray[Any], *args: npt.NDArray[Any]
) -> None: ...

# undocumented
def slice_addition(
    the_slice: slice[int, int, Any], addend: int
) -> slice[int, int, None]: ...

#
class StencilFunc(Generic[_ScalarT_co, Unpack[_Ts]]):
    id_counter: ClassVar[int] = 0

    id: int
    kernel_ir: FunctionIR
    mode: _Mode
    options: _StencilOptions
    kws: list[tuple[str, Any]]
    neighborhood: tuple[tuple[int, int], ...] | None

    def __init__(
        self,
        kernel_ir: FunctionIR,
        mode: _Mode,
        options: _StencilOptions,
    ) -> None: ...

    #
    def replace_return_with_setitem(
        self,
        blocks: Mapping[int, Block],
        index_vars: Sequence[str],
        out_name: str,
    ) -> list[int]: ...

    #
    def add_indices_to_kernel(
        self,
        kernel: FunctionIR,
        index_names: Sequence[str],
        ndim: int,
        neighborhood: tuple[tuple[int, int], ...] | None,
        standard_indexed: Container[str],
        typemap: Mapping[str, types.Type],
        calltypes: Mapping[Expr, Signature],
    ) -> tuple[list[list[int]], set[str]]: ...

    #
    def get_return_type(
        self,
        argtys: tuple[types.Array, *tuple[types.Type, ...]],
    ) -> tuple[
        types.Array,
        Mapping[str, types.Type],
        Mapping[Expr, Signature],
    ]: ...

    #
    def compile_for_argtys(
        self,
        argtys: tuple[types.Array, *tuple[types.Type, ...]],
        kwtys: Mapping[str, types.Type],
        return_type: types.Type,
        sigret: Signature | None,
    ) -> CompileResult: ...

    #
    def copy_ir_with_calltypes(
        self, ir: FunctionIR, calltypes: Mapping[Expr, Signature]
    ) -> tuple[FunctionIR, dict[Expr, Signature]]: ...

    #
    @overload
    def __call__(
        self, /, *args: Unpack[_Ts], out: None = None
    ) -> npt.NDArray[_ScalarT_co]: ...
    @overload
    def __call__(self, /, *args: Unpack[_Ts], out: _ArrayT) -> _ArrayT: ...

#
@overload
def stencil(
    func_or_mode: Callable[[Unpack[_Ts0]], _ScalarT],
    **options: Unpack[_StencilOptions],
) -> StencilFunc[_ScalarT, Unpack[_Ts0]]: ...
@overload  # the mypy error is a false positive due to the invariant return type
def stencil(  # type:ignore[overload-overlap]
    func_or_mode: _InvariantPosonlyCallable[bool, Unpack[_Ts0]],
    **options: Unpack[_StencilOptions],
) -> StencilFunc[np.bool_, Unpack[_Ts0]]: ...
@overload
def stencil(
    func_or_mode: _InvariantPosonlyCallable[int, Unpack[_Ts0]],
    **options: Unpack[_StencilOptions],
) -> StencilFunc[np.int_, Unpack[_Ts0]]: ...
@overload
def stencil(
    func_or_mode: _InvariantPosonlyCallable[float, Unpack[_Ts0]],
    **options: Unpack[_StencilOptions],
) -> StencilFunc[np.float64, Unpack[_Ts0]]: ...
@overload
def stencil(
    func_or_mode: _InvariantPosonlyCallable[complex, Unpack[_Ts0]],
    **options: Unpack[_StencilOptions],
) -> StencilFunc[np.complex128, Unpack[_Ts0]]: ...
@overload
def stencil(
    func_or_mode: Callable[[Unpack[_Ts0]], object],
    **options: Unpack[_StencilOptions],
) -> StencilFunc[Any, Unpack[_Ts0]]: ...
@overload
def stencil(
    func_or_mode: _Mode = "constant",
    **options: Unpack[_StencilOptions],
) -> _StencilWrapper: ...
