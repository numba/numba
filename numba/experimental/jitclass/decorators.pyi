from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias, overload, type_check_only

from typing_extensions import TypeVar

from numba.core import types

###

_ToSpec: TypeAlias = Mapping[str, types.Type] | Sequence[tuple[str, types.Type]]

_ClsT = TypeVar("_ClsT", bound=type)

@type_check_only
class _Wrap(Protocol):
    def __call__(self, cls: _ClsT) -> _ClsT: ...

###

@overload
def jitclass(cls_or_spec: None = None, spec: _ToSpec | None = None) -> _Wrap: ...
@overload
def jitclass(cls_or_spec: _ToSpec, spec: None = None) -> _Wrap: ...
@overload
def jitclass(cls_or_spec: _ClsT, spec: _ToSpec | None = None) -> _ClsT: ...
