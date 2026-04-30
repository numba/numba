# This file is provided by @jorenham with minor modifications (e.g. typos).
# See original content at: https://github.com/numba/numba/pull/9945#pullrequestreview-2668923222.
#
# This file has been tested under:
#   - mypy for the use-case in issue #9900
#   - mypy numba/core/types/__init__.pyi
# Testing with mypy.stubtest does not work due to other mypy errors in the code
# base.
import numpy as np

from .abstract import *
from .common import Opaque
from .containers import *
from .function_type import *
from .functions import *
from .iterators import *
from .misc import *
from .npytypes import *
from .scalars import (
    Boolean,
    BooleanLiteral as BooleanLiteral,
    Complex,
    EnumClass as EnumClass,
    EnumMember as EnumMember,
    Float,
    IntEnumClass as IntEnumClass,
    IntEnumMember as IntEnumMember,
    Integer,
    IntegerLiteral as IntegerLiteral,
    parse_integer_bitwidth as parse_integer_bitwidth,
    parse_integer_signed as parse_integer_signed,
)

__all__ = [
    "b1",
    "bool",  # numpy>=2
    "bool_",
    "boolean",
    "byte",
    "c8",
    "c16",
    "char",
    "complex64",
    "complex128",
    "deferred_type",
    "double",
    "f4",
    "f8",
    "ffi",
    "ffi_forced_object",
    "float32",
    "float64",
    "float_",  # numpy<2
    "i1",
    "i2",
    "i4",
    "i8",
    "int8",
    "int16",
    "int32",
    "int64",
    "int_",
    "intc",
    "intp",
    "long_",
    "longlong",
    "none",
    "optional",
    "short",
    "size_t",
    "ssize_t",
    "u1",
    "u2",
    "u4",
    "u8",
    "uchar",
    "uint",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uintc",
    "uintp",
    "ulong",
    "ulonglong",
    "ushort",
    "void",
]

# TODO: Final

pyobject: PyObject = ...
ffi_forced_object: Opaque = ...
ffi: Opaque = ...
none: NoneType = ...
ellipsis: EllipsisType = ...
Any: Phantom = ...
undefined: Undefined = ...
py2_string_type: Opaque = ...
unicode_type: UnicodeType = ...
string: UnicodeType = ...
unknown: Dummy = ...
npy_rng: NumPyRandomGeneratorType = ...
npy_bitgen: NumPyRandomBitGeneratorType = ...

_undef_var: UndefVar = ...

code_type: Opaque = ...
pyfunc_type: Opaque = ...

voidptr: RawPointer = ...

optional = Optional
deferred_type = DeferredType
slice2_type: SliceType = ...
slice3_type: SliceType = ...
void: NoneType = ...

boolean: Boolean = ...
bool_: Boolean = boolean
bool: Boolean = boolean  # numpy>=2

int8: Integer[np.int8] = ...
int16: Integer[np.int16] = ...
int32: Integer[np.int32] = ...
int64: Integer[np.int64] = ...
intp: Integer[np.intp] = ...
intc: Integer[np.intc] = ...
ssize_t: Integer[np.intp] = ...
char: Integer[np.byte] = ...
short: Integer[np.short] = ...
int_: Integer[np.int_] = ...
long_: Integer[np.int32 | np.int64] = ...
longlong: Integer[np.int64] = ...

byte: Integer[np.ubyte] = ...
uint8: Integer[np.uint8] = ...
uint16: Integer[np.uint16] = ...
uint32: Integer[np.uint32] = ...
uint64: Integer[np.uint64] = ...
uintp: Integer[np.uintp] = ...
uintc: Integer[np.uintc] = ...
size_t: Integer[np.uintp] = ...
uchar: Integer[np.ubyte] = ...
ushort: Integer[np.ushort] = ...
uint: Integer[np.uint] = ...
ulong: Integer[np.uint32 | np.uint64] = ...
ulonglong: Integer[np.uint64] = ...

float16: Float[np.float16] = ...
float32: Float[np.float32] = ...
float64: Float[np.float64] = ...
float_: Float[np.float32] = float32  # numpy<2
double: Float[np.float64] = float64

complex64: Complex[np.complex64, np.float32] = ...
complex128: Complex[np.complex128, np.float64] = ...

range_iter32_type: RangeIteratorType = ...
range_iter64_type: RangeIteratorType = ...
unsigned_range_iter64_type: RangeIteratorType = ...
range_state32_type: RangeType = ...
range_state64_type: RangeType = ...
unsigned_range_state64_type: RangeType = ...

signed_domain: frozenset[Integer[np.signedinteger]] = ...
unsigned_domain: frozenset[Integer[np.unsignedinteger]] = ...
integer_domain: frozenset[Integer] = ...
real_domain: frozenset[Float] = ...
complex_domain: frozenset[Complex] = ...
number_domain: frozenset[Integer | Float | Complex] = ...

c_bool: Boolean = boolean
c_int8: Integer[np.int8] = int8
c_int16: Integer[np.int16] = int16
c_int32: Integer[np.int32] = int32
c_int64: Integer[np.int64] = int64
c_intp: Integer[np.intp] = intp
c_uint8: Integer[np.uint8] = uint8
c_uint16: Integer[np.uint16] = uint16
c_uint32: Integer[np.uint32] = uint32
c_uint64: Integer[np.uint64] = uint64
c_uintp: Integer[np.uintp] = uintp
c_float16: Float[np.float16] = float16
c_float32: Float[np.float32] = float32
c_float64: Float[np.float64] = float64

np_bool_: Boolean = boolean
np_int8: Integer[np.int8] = int8
np_int16: Integer[np.int16] = int16
np_int32: Integer[np.int32] = int32
np_int64: Integer[np.int64] = int64
np_intp: Integer[np.intp] = intp
np_uint8: Integer[np.uint8] = uint8
np_uint16: Integer[np.uint16] = uint16
np_uint32: Integer[np.uint32] = uint32
np_uint64: Integer[np.uint64] = uint64
np_uintp: Integer[np.uintp] = uintp
np_float16: Float[np.float16] = float16
np_float32: Float[np.float32] = float32
np_float64: Float[np.float64] = float64
np_float_: Float[np.float32] = float32
np_double: Float[np.float64] = float64
np_complex64: Complex[np.complex64, np.float32] = complex64
np_complex128: Complex[np.complex128, np.float64] = complex128

py_bool: Boolean = boolean
py_int: Integer[np.intp] = intp
py_float: Float[np.float64] = float64
py_complex: Complex[np.complex128, np.float64] = complex128

py_signed_domain: frozenset[Integer[np.signedinteger]] = signed_domain
py_integer_domain: frozenset[Integer] = integer_domain
py_real_domain: frozenset[Float] = real_domain
py_complex_domain: frozenset[Complex] = complex_domain
py_number_domain: frozenset[Integer | Float | Complex] = number_domain

np_signed_domain: frozenset[Integer[np.signedinteger]] = signed_domain
np_unsigned_domain: frozenset[Integer[np.unsignedinteger]] = unsigned_domain
np_integer_domain: frozenset[Integer] = integer_domain
np_real_domain: frozenset[Float] = real_domain
np_complex_domain: frozenset[Complex] = complex_domain
np_number_domain: frozenset[Integer | Float | Complex] = number_domain

b1: Boolean = bool_
i1: Integer[np.int8] = int8
i2: Integer[np.int16] = int16
i4: Integer[np.int32] = int32
i8: Integer[np.int64] = int64
u1: Integer[np.uint8] = uint8
u2: Integer[np.uint16] = uint16
u4: Integer[np.uint32] = uint32
u8: Integer[np.uint64] = uint64
f2: Float[np.float16] = float16
f4: Float[np.float32] = float32
f8: Float[np.float64] = float64
c8: Complex[np.complex64, np.float32] = complex64
c16: Complex[np.complex128, np.float64] = complex128
