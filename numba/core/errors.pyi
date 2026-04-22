from collections.abc import Callable, Iterable
from typing import Any, Final, Protocol, TypeAlias

from typing_extensions import Generic, Self, TypeVar, override

from .ir import Loc
from .target_extension import Target
from .types import Type

__all__ = [
    "ByteCodeSupportError",
    "CompilerError",
    "ConstantInferenceError",
    "DeprecationError",
    "ForbiddenConstruct",
    "ForceLiteralArg",
    "IRError",
    "InternalError",
    "InternalTargetMismatchError",
    "LiteralTypingError",
    "LoweringError",
    "NonexistentTargetError",
    "NotDefinedError",
    "NumbaAssertionError",
    "NumbaAttributeError",
    "NumbaDebugInfoWarning",
    "NumbaDeprecationWarning",
    "NumbaError",
    "NumbaExperimentalFeatureWarning",
    "NumbaIRAssumptionWarning",
    "NumbaIndexError",
    "NumbaInvalidConfigWarning",
    "NumbaKeyError",
    "NumbaNotImplementedError",
    "NumbaParallelSafetyWarning",
    "NumbaPedanticWarning",
    "NumbaPendingDeprecationWarning",
    "NumbaPerformanceWarning",
    "NumbaRuntimeError",
    "NumbaSystemWarning",
    "NumbaTypeError",
    "NumbaTypeSafetyWarning",
    "NumbaValueError",
    "NumbaWarning",
    "RedefinedError",
    "RequireLiteralValue",
    "TypingError",
    "UnsupportedBytecodeError",
    "UnsupportedError",
    "UnsupportedParforsError",
    "UnsupportedRewriteError",
    "UntypedAttributeError",
    "VerificationError",
]

_ExceptionT_co = TypeVar(
    "_ExceptionT_co",
    bound=BaseException | str,
    default=Any,
    covariant=True,
)

_Ignored: TypeAlias = object
_FoldArgumentsFn: TypeAlias = Callable[
    [tuple[Any, ...], dict[str, Any]],
    tuple[Any, ...],
]

###

class NumbaWarning(Warning):
    msg: str
    loc: Loc | None

    @override
    def __init__(
        self,
        msg: str,
        loc: Loc | None = None,
        highlighting: bool = True,
    ) -> None: ...

class NumbaPerformanceWarning(NumbaWarning): ...
class NumbaDeprecationWarning(NumbaWarning, DeprecationWarning): ...
class NumbaPendingDeprecationWarning(NumbaWarning, PendingDeprecationWarning): ...
class NumbaParallelSafetyWarning(NumbaWarning): ...
class NumbaTypeSafetyWarning(NumbaWarning): ...
class NumbaExperimentalFeatureWarning(NumbaWarning): ...
class NumbaInvalidConfigWarning(NumbaWarning): ...

class NumbaPedanticWarning(NumbaWarning):
    @override
    def __init__(self, msg: str) -> None: ...

class NumbaIRAssumptionWarning(NumbaPedanticWarning): ...
class NumbaDebugInfoWarning(NumbaWarning): ...
class NumbaSystemWarning(NumbaWarning): ...

class NumbaError(Exception):
    @override
    def __init__(
        self,
        msg: str,
        loc: Loc | None = None,
        highlighting: bool = True,
    ) -> None: ...
    @property
    def contexts(self) -> list[str]: ...
    def add_context(self, msg: str) -> Self: ...
    def patch_message(self, new_message: str) -> None: ...

class UnsupportedError(NumbaError): ...

class UnsupportedBytecodeError(Exception):
    @override
    def __init__(self, msg: str, loc: Loc | None = None) -> None: ...

class UnsupportedRewriteError(UnsupportedError): ...
class IRError(NumbaError): ...
class RedefinedError(IRError): ...

class NotDefinedError(IRError):
    @override
    def __init__(self, name: str, loc: Loc | None = None) -> None: ...

class VerificationError(IRError): ...
class DeprecationError(NumbaError): ...

class LoweringError(NumbaError):
    @override
    def __init__(self, msg: str, loc: Loc | None = None) -> None: ...

class UnsupportedParforsError(NumbaError): ...
class ForbiddenConstruct(LoweringError): ...
class TypingError(NumbaError): ...

class UntypedAttributeError(TypingError):
    @override
    def __init__(self, value: Type, attr: str, loc: Loc | None = None) -> None: ...

class ByteCodeSupportError(NumbaError):
    @override
    def __init__(self, msg: str, loc: Loc | None = None) -> None: ...

class CompilerError(NumbaError): ...

class ConstantInferenceError(NumbaError):
    @override
    def __init__(self, value: str, loc: Loc | None = None) -> None: ...

class InternalError(NumbaError, Generic[_ExceptionT_co]):
    old_exception: _ExceptionT_co

    @override
    def __init__(self, exception: _ExceptionT_co) -> None: ...

class InternalTargetMismatchError(InternalError[str]):
    @override
    def __init__(self, kind: str, target_hw: str, hw_clazz: type[Target]) -> None: ...

class NonexistentTargetError(InternalError[_ExceptionT_co]): ...
class RequireLiteralValue(TypingError): ...

class ForceLiteralArg(NumbaError):
    requested_args: Final[frozenset[int]]
    fold_arguments: Final[_FoldArgumentsFn | None]

    @override
    def __init__(
        self,
        arg_indices: Iterable[int],
        fold_arguments: _FoldArgumentsFn | None = None,
        loc: Loc | None = None,
    ) -> None: ...

    #
    def bind_fold_arguments(
        self,
        fold_arguments: _FoldArgumentsFn,
    ) -> ForceLiteralArg: ...

    #
    def combine(self, other: ForceLiteralArg) -> ForceLiteralArg: ...
    def __or__(self, other: ForceLiteralArg) -> ForceLiteralArg: ...

class LiteralTypingError(TypingError): ...
class NumbaValueError(TypingError): ...
class NumbaTypeError(TypingError): ...
class NumbaAttributeError(TypingError): ...
class NumbaAssertionError(TypingError): ...
class NumbaNotImplementedError(TypingError): ...
class NumbaKeyError(TypingError): ...
class NumbaIndexError(TypingError): ...
class NumbaRuntimeError(NumbaError): ...

class _SupportsColorScheme(Protocol):
    def __init__(self, theme: str | None = None) -> None: ...
    def code(self, msg: str) -> str: ...
    def errmsg(self, msg: str) -> str: ...
    def filename(self, msg: str) -> str: ...
    def indicate(self, msg: str) -> str: ...
    def highlight(self, msg: str) -> str: ...
    def reset(self, msg: str) -> str: ...

def termcolor() -> _SupportsColorScheme: ...
