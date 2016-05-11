from __future__ import print_function, division, absolute_import

import struct

import numpy as np

from .abstract import *
from .containers import *
from .functions import *
from .iterators import *
from .misc import *
from .npytypes import *
from .scalars import *


# Short names

pyobject = PyObject('pyobject')
ffi_forced_object = Opaque('ffi_forced_object')
ffi = Opaque('ffi')
none = NoneType('none')
ellipsis = EllipsisType('...')
Any = Phantom('any')
undefined = Undefined('undefined')
string = Opaque('str')

# No operation is defined on voidptr
# Can only pass it around
voidptr = RawPointer('void*')

boolean = bool_ = Boolean('bool')

byte = uint8 = Integer('uint8')
uint16 = Integer('uint16')
uint32 = Integer('uint32')
uint64 = Integer('uint64')

int8 = Integer('int8')
int16 = Integer('int16')
int32 = Integer('int32')
int64 = Integer('int64')
intp = int32 if utils.MACHINE_BITS == 32 else int64
uintp = uint32 if utils.MACHINE_BITS == 32 else uint64
intc = int32 if struct.calcsize('i') == 4 else int64
uintc = uint32 if struct.calcsize('i') == 4 else uint64

float32 = Float('float32')
float64 = Float('float64')

complex64 = Complex('complex64', float32)
complex128 = Complex('complex128', float64)

range_iter32_type = RangeIteratorType(int32)
range_iter64_type = RangeIteratorType(int64)
unsigned_range_iter64_type = RangeIteratorType(uint64)
range_state32_type = RangeType(int32)
range_state64_type = RangeType(int64)
unsigned_range_state64_type = RangeType(uint64)

slice2_type = SliceType('slice<a:b>', 2)
slice3_type = SliceType('slice<a:b:c>', 3)

signed_domain = frozenset([int8, int16, int32, int64])
unsigned_domain = frozenset([uint8, uint16, uint32, uint64])
integer_domain = signed_domain | unsigned_domain
real_domain = frozenset([float32, float64])
complex_domain = frozenset([complex64, complex128])
number_domain = real_domain | integer_domain | complex_domain

# Aliases to Numpy type names

b1 = bool_
i1 = int8
i2 = int16
i4 = int32
i8 = int64
u1 = uint8
u2 = uint16
u4 = uint32
u8 = uint64

f4 = float32
f8 = float64

c8 = complex64
c16 = complex128

float_ = float32
double = float64
void = none

_make_signed = lambda x: globals()["int%d" % (np.dtype(x).itemsize * 8)]
_make_unsigned = lambda x: globals()["uint%d" % (np.dtype(x).itemsize * 8)]

char = _make_signed(np.byte)
uchar = byte = _make_unsigned(np.byte)
short = _make_signed(np.short)
ushort = _make_unsigned(np.short)
int_ = _make_signed(np.int_)
uint = _make_unsigned(np.int_)
intc = _make_signed(np.intc) # C-compat int
uintc = _make_unsigned(np.uintc) # C-compat uint
long_ = _make_signed(np.long)
ulong = _make_unsigned(np.long)
longlong = _make_signed(np.longlong)
ulonglong = _make_unsigned(np.longlong)

# optional types
optional = Optional


deferred_type = DeferredType

__all__ = '''
int8
int16
int32
int64
uint8
uint16
uint32
uint64
intp
uintp
intc
uintc
boolean
float32
float64
complex64
complex128
bool_
byte
char
uchar
short
ushort
int_
uint
long_
ulong
longlong
ulonglong
float_
double
void
none
b1
i1
i2
i4
i8
u1
u2
u4
u8
f4
f8
c8
c16
optional
ffi_forced_object
ffi
deferred_type
'''.split()
