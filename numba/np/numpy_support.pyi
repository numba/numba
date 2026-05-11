import ctypes as ct
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final, Literal, NamedTuple, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeIs

from numba.core import types, typing
from numba.np import types as npy_types

from numba.core.cgutils import is_nonelike as is_nonelike

###

_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])

_ToDType: TypeAlias = np.dtype[_ScalarT] | type[_ScalarT]
_CastingKind: TypeAlias = Literal[
    "no", "equiv", "safe", "same_kind", "same_value", "unsafe"
]

###

numpy_version: Final[tuple[int, int]] = ...  # undocumented
FROM_DTYPE: Final[Mapping[np.dtype[Any], types.Type]] = ...  # undocumented

re_typestr: Final[re.Pattern[str]] = ...  # undocumented
re_datetimestr: Final[re.Pattern[str]] = ...  # undocumented

# undocumented
def as_dtype(nbtype: types.Type) -> np.dtype[Any]: ...

# undocumented
def as_struct_dtype(rec: types.Record) -> np.dtype[np.void]: ...

# undocumented
@overload
def map_arrayscalar_type(val: np.bool_ | bool) -> types.Boolean: ...
@overload
def map_arrayscalar_type(val: np.integer[Any]) -> types.Integer: ...
@overload
def map_arrayscalar_type(val: int) -> types.Integer | Any: ...
@overload
def map_arrayscalar_type(val: np.floating[Any]) -> types.Float: ...
@overload
def map_arrayscalar_type(val: float) -> types.Float | Any: ...
@overload
def map_arrayscalar_type(val: np.complexfloating[Any, Any]) -> types.Complex: ...
@overload
def map_arrayscalar_type(val: complex) -> types.Complex | Any: ...
@overload
def map_arrayscalar_type(val: np.datetime64) -> npy_types.NPDatetime: ...
@overload
def map_arrayscalar_type(val: np.timedelta64) -> npy_types.NPTimedelta: ...

# undocumented
def is_array(val: object) -> TypeIs[npt.NDArray[Any]]: ...

# undocumented
def map_layout(val: npt.NDArray[Any] | np.generic) -> Literal["C", "F", "A"]: ...

# undocumented
def select_array_wrapper(inputs: Iterable[types.ArrayCompatible | Any]) -> int: ...

# undocumented
def resolve_output_type(
    context: typing.BaseContext,
    inputs: Sequence[types.ArrayCompatible | Any],
    formal_output: types.Array,
) -> types.Type: ...

# undocumented
def supported_ufunc_loop(ufunc: np.ufunc, loop: UFuncLoopSpec) -> bool: ...

# undocumented
class UFuncLoopSpec(NamedTuple):
    inputs: Iterable[types.Type]
    outputs: Iterable[types.Type]
    ufunc_sig: str

    @property
    def numpy_inputs(self) -> list[np.dtype[Any]]: ...
    @property
    def numpy_outputs(self) -> list[np.dtype[Any]]: ...

# undocumented
def ufunc_can_cast(
    from_: npt.DTypeLike,
    to: npt.DTypeLike,
    has_mixed_inputs: bool,
    casting: _CastingKind = "safe",
) -> bool: ...

# undocumented
def ufunc_find_matching_loop(
    ufunc: np.ufunc,
    arg_types: tuple[types.Type, ...],
) -> UFuncLoopSpec | None: ...

# undocumented
def from_struct_dtype(dtype: np.dtype[np.void]) -> types.Record: ...

# undocumented
def is_contiguous(
    dims: tuple[int, ...], strides: tuple[int, ...], itemsize: int
) -> bool: ...

# undocumented
def is_fortran(
    dims: tuple[int, ...], strides: tuple[int, ...], itemsize: int
) -> bool: ...

# undocumented
def type_can_asarray(
    arr: object,
) -> TypeIs[
    types.Array
    | types.Sequence
    | types.Tuple
    | types.StringLiteral
    | types.Number
    | types.Boolean
    | types.containers.ListType
]: ...

# undocumented
def type_is_scalar(
    typ: object,
) -> TypeIs[
    types.Boolean
    | types.Number
    | types.UnicodeType
    | types.StringLiteral
    | npy_types.NPTimedelta
    | npy_types.NPDatetime
]: ...

# undocumented
def check_is_integer(v: int | types.Integer, name: str) -> None: ...

# undocumented
def lt_floats(a: float | np.floating, b: float | np.floating) -> bool: ...

# undocumented
def lt_complex(
    a: complex | np.complexfloating, b: complex | np.complexfloating
) -> bool: ...

#
@overload
def from_dtype(dtype: _ToDType[np.bool_]) -> types.Boolean: ...
@overload
def from_dtype(dtype: _ToDType[np.integer[Any]]) -> types.Integer: ...
@overload
def from_dtype(dtype: _ToDType[np.floating[Any]]) -> types.Float: ...
@overload
def from_dtype(dtype: _ToDType[np.complexfloating[Any, Any]]) -> types.Complex: ...
@overload
def from_dtype(dtype: _ToDType[np.object_]) -> types.PyObject: ...
@overload
def from_dtype(dtype: _ToDType[np.bytes_]) -> types.CharSeq: ...
@overload
def from_dtype(dtype: _ToDType[np.str_]) -> types.UnicodeCharSeq: ...
@overload
def from_dtype(dtype: _ToDType[np.datetime64]) -> npy_types.NPDatetime: ...
@overload
def from_dtype(dtype: _ToDType[np.timedelta64]) -> npy_types.NPTimedelta: ...
@overload
def from_dtype(dtype: _ToDType[np.void]) -> types.Record | types.NestedArray: ...

#
@overload
def carray(
    ptr: ct._Pointer[Any], shape: int, dtype: None = None
) -> np.ndarray[tuple[int], np.dtype[Any]]: ...
@overload
def carray(
    ptr: ct._Pointer[Any], shape: _ShapeT, dtype: None = None
) -> np.ndarray[_ShapeT, np.dtype[Any]]: ...
@overload
def carray(
    ptr: ct.c_void_p | ct._Pointer[Any], shape: int, dtype: _ToDType[_ScalarT]
) -> np.ndarray[tuple[int], np.dtype[_ScalarT]]: ...
@overload
def carray(
    ptr: ct.c_void_p | ct._Pointer[Any], shape: _ShapeT, dtype: _ToDType[_ScalarT]
) -> np.ndarray[_ShapeT, np.dtype[_ScalarT]]: ...

# keep in sync with `carray`
@overload
def farray(
    ptr: ct._Pointer[Any], shape: int, dtype: None = None
) -> np.ndarray[tuple[int], np.dtype[Any]]: ...
@overload
def farray(
    ptr: ct._Pointer[Any], shape: _ShapeT, dtype: None = None
) -> np.ndarray[_ShapeT, np.dtype[Any]]: ...
@overload
def farray(
    ptr: ct.c_void_p | ct._Pointer[Any], shape: int, dtype: _ToDType[_ScalarT]
) -> np.ndarray[tuple[int], np.dtype[_ScalarT]]: ...
@overload
def farray(
    ptr: ct.c_void_p | ct._Pointer[Any], shape: _ShapeT, dtype: _ToDType[_ScalarT]
) -> np.ndarray[_ShapeT, np.dtype[_ScalarT]]: ...
