from builtins import range as prange
from typing import TypeVar

from _typeshed import Incomplete
from numpy import ndindex as pndindex

from numba.core.typing.asnumbatype import as_numba_type
from numba.core.typing.typeof import typeof

__all__ = [
    "typeof",
    "as_numba_type",
    "prange",
    "pndindex",
    "gdb",
    "gdb_breakpoint",
    "gdb_init",
    "literally",
    "literal_unroll",
]

###

_T = TypeVar("_T")

###

def gdb(*args: Incomplete) -> None: ...
def gdb_breakpoint() -> None: ...
def gdb_init(*args: Incomplete) -> None: ...

#
def literally(obj: _T) -> _T: ...
def literal_unroll(container: _T) -> _T: ...
