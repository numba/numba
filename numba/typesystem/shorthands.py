"""
Shorthands for type constructing, promotions, etc.
"""

from numba.typesystem import *
from numba.minivect import minitypes

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

def get_type(ast_node):
    """
    :param ast_node: a Numba or Python AST expression node
    :return: the type of the expression node
    """
    return ast_node.variable.type

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

#------------------------------------------------------------------------
# Type Constructor Shorthands
#------------------------------------------------------------------------

def from_numpy_dtype(np_dtype):
    """
    :param np_dtype: the NumPy dtype (e.g. np.dtype(np.double))
    :return: a dtype type representation
    """
    return dtype(minitypes.map_dtype(np_dtype))

def dtype(dtype_type):
    """

    :param dtype: the Numba dtype type (e.g. double)
    :return: a dtype type representation
    """
    assert isinstance(dtype_type, minitypes.Type)
    return NumpyDtypeType(dtype_type)

def array(dtype, ndim):
    """
    :param dtype: the Numba dtype type (e.g. double)
    :param ndim: the array dimensionality (int)
    :return: an array type representation
    """
    if ndim == 0:
        return dtype
    return minitypes.ArrayType(dtype, ndim)

def tuple_(base_type, size=-1):
    """
    :param base_type: the element type of the tuple
    :param size: set to a value >= 0 is the size is known
    :return: a tuple type representation
    """
    return TupleType(base_type, size)

def list_(base_type, size=-1):
    """
    :param base_type: the element type of the tuple
    :param size: set to a value >= 0 is the size is known
    :return: a tuple type representation
    """
    return ListType(base_type, size)
