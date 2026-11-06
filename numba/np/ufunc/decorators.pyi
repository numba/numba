from collections.abc import Callable, Iterable
from typing import Any, ClassVar, Literal, TypeAlias, overload

import numpy as np

from numba.core.registry import DelayedRegistry
from numba.core.types import Type
from numba.core.typing.templates import Signature
from numba.cuda.vectorizers import (
    CUDAGeneralizedUFunc,
    CUDAGUFuncVectorize,
    CUDAUFuncDispatcher,
    CUDAVectorize,
)
from numba.np.ufunc import dufunc, gufunc
from numba.np.ufunc.parallel import ParallelGUFuncBuilder, ParallelUFuncBuilder

###

_Identity: TypeAlias = Literal[0, 1, "reorderable"]
_ToSignature: TypeAlias = Signature | tuple[Type | Signature, ...] | str

###

class _BaseVectorize:
    target_registry: ClassVar[DelayedRegistry]  # abstract

    @classmethod
    def get_identity(cls, kwargs: dict[str, Any]) -> Any | None: ...
    @classmethod
    def get_cache(cls, kwargs: dict[str, Any]) -> bool: ...
    @classmethod
    def get_writable_args(cls, kwargs: dict[str, Any]) -> tuple[int, ...]: ...
    @classmethod
    def get_target_implementation(cls, kwargs: dict[str, Any]) -> type: ...

class Vectorize(_BaseVectorize):
    @overload
    def __new__(
        cls,
        func: Callable[..., object],
        *,
        target: Literal["cpu"] = "cpu",
        identity: _Identity | None = None,
        cache: bool = False,
        **targetoptions: Any,
    ) -> dufunc.DUFunc: ...
    @overload
    def __new__(
        cls,
        func: Callable[..., object],
        *,
        target: Literal["parallel"],
        identity: _Identity | None = None,
        cache: bool = False,
        **targetoptions: Any,
    ) -> ParallelUFuncBuilder: ...
    @overload
    def __new__(
        cls,
        func: Callable[..., object],
        *,
        target: Literal["cuda"],
        identity: _Identity | None = None,
        cache: Literal[False] = False,
        nopython: bool = True,
    ) -> CUDAVectorize: ...

class GUVectorize(_BaseVectorize):
    @overload
    def __new__(
        cls,
        func: Callable[..., object],
        signature: str,
        *,
        target: Literal["cpu"] = "cpu",
        identity: _Identity | None = None,
        cache: bool = False,
        writable_args: tuple[int, ...] = (),
        is_dynamic: bool = False,
        **targetoptions: Any,
    ) -> gufunc.GUFunc: ...
    @overload
    def __new__(
        cls,
        func: Callable[..., object],
        signature: str,
        *,
        target: Literal["parallel"],
        identity: _Identity | None = None,
        cache: bool = False,
        writable_args: tuple[int, ...] = (),
        **targetoptions: Any,
    ) -> ParallelGUFuncBuilder: ...
    @overload
    def __new__(
        cls,
        func: Callable[..., object],
        signature: str,
        *,
        target: Literal["cuda"],
        identity: _Identity | None = None,
        cache: Literal[False] = False,
        writable_args: tuple[()] = (),
        nopython: Literal[True] = True,
    ) -> CUDAGUFuncVectorize: ...

#
@overload  # function
def vectorize(
    ftylist_or_function: Callable[..., object],
    *,
    identity: _Identity | None = None,
    cache: bool = False,
    targetoptions: dict[str, Any] | None = None,
) -> dufunc.DUFunc: ...
@overload  # ftylist, target="cpu" (default)
def vectorize(
    ftylist_or_function: str | Iterable[_ToSignature] = (),
    *,
    target: Literal["cpu"] = "cpu",
    identity: _Identity | None = None,
    cache: bool = False,
    **targetoptions: Any,
) -> Callable[[Callable[..., Any]], dufunc.DUFunc]: ...
@overload  # ftylist, target="parallel"
def vectorize(
    ftylist_or_function: str | Iterable[_ToSignature] = (),
    *,
    target: Literal["parallel"],
    identity: _Identity | None = None,
    cache: bool = False,
    **targetoptions: Any,
) -> Callable[[Callable[..., Any]], np.ufunc]: ...
@overload  # ftylist, target="cuda"
def vectorize(
    ftylist_or_function: str | Iterable[_ToSignature] = (),
    *,
    target: Literal["cuda"],
    identity: _Identity | None = None,
    cache: Literal[False] = False,
    nopython: bool = True,
) -> Callable[[Callable[..., Any]], CUDAUFuncDispatcher]: ...

#
@overload  # signature, target="cpu"  (default)
def guvectorize(
    signature: str,
    /,
    *,
    target: Literal["cpu"] = "cpu",
    identity: _Identity | None = None,
    cache: bool = False,
    writable_args: tuple[int, ...] = (),
    is_dynamic: bool = True,
    **targetoptions: Any,
) -> Callable[[Callable[..., Any]], gufunc.GUFunc]: ...
@overload  # ftylist, signature, target="cpu"  (default)
def guvectorize(
    ftylist: str | Iterable[_ToSignature],
    signature: str,
    /,
    *,
    target: Literal["cpu"] = "cpu",
    identity: _Identity | None = None,
    cache: bool = False,
    writable_args: tuple[int, ...] = (),
    is_dynamic: bool = False,
    **targetoptions: Any,
) -> Callable[[Callable[..., Any]], gufunc.GUFunc]: ...
@overload  # ftylist, signature, target="parallel"
def guvectorize(
    ftylist: str | Iterable[_ToSignature],
    signature: str,
    /,
    *,
    target: Literal["parallel"],
    identity: _Identity | None = None,
    cache: bool = False,
    writable_args: tuple[int, ...] = (),
    **targetoptions: Any,
) -> Callable[[Callable[..., Any]], np.ufunc]: ...
@overload  # ftylist, signature, target="cuda"
def guvectorize(
    ftylist: str | Iterable[_ToSignature],
    signature: str,
    /,
    *,
    target: Literal["cuda"],
    identity: _Identity | None = None,
    cache: Literal[False] = False,
    writable_args: tuple[()] = (),
    nopython: Literal[True] = True,
) -> Callable[[Callable[..., Any]], CUDAGeneralizedUFunc]: ...
