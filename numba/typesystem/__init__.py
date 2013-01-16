from basetypes import *
from exttypes import *
from closuretypes import *
from ssatypes import *
from templatetypes import *
from typemapper import *

__all__ = minitypes.__all__ + [
    'O', 'b1', 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8',
    'f4', 'f8', 'f16', 'c8', 'c16', 'c32', 'template',
]

#------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------

def is_obj(type):
    return type.is_object or type.is_array

native_type_dict = {}
for native_type in minitypes.native_integral:
    native_type_dict[(native_type.itemsize, native_type.signed)] = native_type

def promote_to_native(int_type):
    return native_type_dict[int_type.itemsize, int_type.signed]

def promote_closest(context, int_type, candidates):
    """
    promote_closest(Py_ssize_t, [int_, long_, longlong]) -> longlong
    """
    for candidate in candidates:
        promoted = context.promote_types(int_type, candidate)
        if promoted.itemsize == candidate.itemsize and promoted.signed == candidate.signed:
            return candidate

    return candidates[-1]

#------------------------------------------------------------------------
# Type shorthands
#------------------------------------------------------------------------

O = object_
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
f16 = float128

c8 = complex64
c16 = complex128
c32 = complex256
