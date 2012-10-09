import math
import copy
import types
import ctypes.util

import llvm.core
import numpy as np
from numpy import ctypeslib
# from numpy.ctypeslib import _typecodes

from numba import llvm_types
from numba.minivect.minitypes import *
from numba.minivect import miniast, minitypes
from numba.minivect.ctypes_conversion import convert_to_ctypes, convert_from_ctypes
from numba.minivect.complex_support import Complex64, Complex128, Complex256

__all__ = minitypes.__all__ + [
    'O', 'b1', 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8',
    'f4', 'f8', 'f16', 'c8', 'c16', 'c32' 
]

class NumbaType(minitypes.Type):
    is_numba_type = True

class TupleType(NumbaType, minitypes.ObjectType):
    is_tuple = True
    name = "tuple"
    size = 0

    def __str__(self):
        return "tuple(%s)" % ", ".join(["..."] * self.size)

class ListType(NumbaType, minitypes.ObjectType):
    is_list = True
    name = "list"
    size = 0

    def __str__(self):
        return "list(%s)" % ", ".join(["..."] * self.size)

class IteratorType(NumbaType, minitypes.ObjectType):
    is_iterator = True
    subtypes = ['base_type']

    def __init__(self, base_type, **kwds):
        super(IteratorType, self).__init__(**kwds)
        self.base_type = base_type

    def __repr__(self):
        return "iterator<%s>" % (self.base_type,)

class PHIType(NumbaType):
    """
    Type for phi() values.
    """
    is_phi = True

class ModuleType(NumbaType, minitypes.ObjectType):
    """
    Represents a type for modules.

    Attributes:
        is_numpy_module: whether the module is the numpy module
        module: in case of numpy, the numpy module or a submodule
    """
    is_module = True
    is_numpy_module = False
    module = None

    def __repr__(self):
        if self.is_numpy_module:
            return 'numpy'
        else:
            return 'ModuleType'

class NumpyAttributeType(NumbaType, minitypes.ObjectType):
    """
    Type for attributes of a numpy (sub)module.

    Attributes:
        module: the numpy (sub)module
        attr: the attribute name (str)
    """
    is_numpy_attribute = True
    module = None
    attr = None

    def __repr__(self):
        return "%s.%s" % (self.module.__name__, self.attr)

    @property
    def value(self):
        return getattr(self.module, self.attr)

class MethodType(NumbaType, minitypes.ObjectType):
    """
    Method of something
    """

    is_method = True

    def __init__(self, base_type, attr_name, **kwds):
        super(MethodType, self).__init__(**kwds)
        self.base_type = base_type
        self.attr_name = attr_name

class NumpyDtypeType(NumbaType, minitypes.ObjectType):
    is_numpy_dtype = True
    dtype = None

    def resolve(self):
        return _map_dtype(self.dtype)

class EllipsisType(NumbaType, minitypes.ObjectType):
    is_ellipsis = True

    def __eq__(self, other):
        return other.is_ellipsis

    def __repr__(self):
        return "..."

class SliceType(NumbaType, minitypes.ObjectType):
    is_slice = True

    def __eq__(self, other):
        return other.is_slice

    def __repr__(self):
        return ":"

class NewAxisType(NumbaType, minitypes.ObjectType):
    is_newaxis = True

    def __eq__(self, other):
        return other.is_newaxis

    def __repr__(self):
        return "newaxis"

class GlobalType(NumbaType, minitypes.ObjectType):
    is_global = True

class BuiltinType(NumbaType, minitypes.ObjectType):
    is_builtin = True

class ModuleAttributeType(NumbaType, minitypes.ObjectType):
    is_module_attr = True

class RangeType(NumbaType):
    is_range = True

class NoneType(NumbaType, minitypes.ObjectType):
    is_none = True

    def __str__(self):
        return "None Type"

class CTypesFunctionType(NumbaType, minitypes.ObjectType):
    is_ctypes_function = True

    def __init__(self, ctypes_func, restype, argtypes, **kwds):
        super(CTypesFunctionType, self).__init__(**kwds)
        self.ctypes_func = ctypes_func
        self.signature = minitypes.FunctionType(return_type=restype,
                                                args=argtypes)

    def __str__(self):
        return "<ctypes function %s>" % (self.signature,)


tuple_ = TupleType()
phi = PHIType()
module_type = ModuleType()
none = NoneType()

intp = minitypes.npy_intp

#
### Type shorthands
#

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

f4 = float_
f8 = double
f16 = float128

c8 = complex64
c16 = complex128
c32 = complex256

class NumbaTypeMapper(minitypes.TypeMapper):


    def __init__(self, context):
        super(NumbaTypeMapper, self).__init__(context)
        # self.ctypes_func_type = type(ctypes.CFUNCTYPE(ctypes.c_int))
        # libc = ctypes.CDLL(ctypes.util.find_library('c'))
        # self.ctypes_func_type2 = type(libc.printf)

    def to_llvm(self, type):
        if type.is_array:
            return llvm_types._numpy_array
        elif type.is_complex:
            lbase_type = type.base_type.to_llvm(self.context)
            return llvm.core.Type.struct([lbase_type, lbase_type])
        elif type.is_py_ssize_t:
            return llvm_types._llvm_py_ssize_t
        elif type.is_object:
            return llvm_types._pyobject_head_struct_p

        return super(NumbaTypeMapper, self).to_llvm(type)

    def from_python(self, value):
        if isinstance(value, np.ndarray):
            dtype = _map_dtype(value.dtype)
            return minitypes.ArrayType(dtype, value.ndim,
                                       is_c_contig=value.flags['C_CONTIGUOUS'],
                                       is_f_contig=value.flags['F_CONTIGUOUS'])
        elif isinstance(value, tuple):
            return tuple_
        elif isinstance(value, types.ModuleType):
            return module_type
        # elif isinstance(value, (self.ctypes_func_type, self.ctypes_func_type2)):
        elif hasattr(value, 'errcheck'):
            # ugh, ctypes
            if value.argtypes is None:
                return object_

            restype = convert_from_ctypes(value.restype)
            argtypes = map(convert_from_ctypes, value.argtypes)
            return CTypesFunctionType(value, restype, argtypes)
        else:
            return super(NumbaTypeMapper, self).from_python(value)

    def promote_types(self, type1, type2):
        if (type1.is_array or type2.is_array) and not \
            (type1.is_array and type2.is_array):
            if type1.is_array:
                array_type = type1
                other_type = type2
            else:
                array_type = type2
                other_type = type1

            type = copy.copy(array_type)
            type.dtype = self.promote_types(array_type.dtype, other_type)
            return type

        return super(NumbaTypeMapper, self).promote_types(type1, type2)


def _map_dtype(dtype):
    """
    >>> _map_dtype(np.dtype(np.int32))
    int32
    >>> _map_dtype(np.dtype(np.int64))
    int64
    >>> _map_dtype(np.dtype(np.object))
    PyObject *
    >>> _map_dtype(np.dtype(np.float64))
    double
    >>> _map_dtype(np.dtype(np.complex128))
    complex128
    """
    item_idx = int(math.log(dtype.itemsize, 2))
    if dtype.kind == 'i':
        return [i1, i2, i4, i8][item_idx]
    elif dtype.kind == 'u':
        return [u1, u2, u4, u8][item_idx]
    elif dtype.kind == 'f':
        if dtype.itemsize == 2:
            pass # half floats not supported yet
        elif dtype.itemsize == 4:
            return f4
        elif dtype.itemsize == 8:
            return f8
        elif dtype.itemsize == 16:
            return f16
    elif dtype.kind == 'b':
        return i1
    elif dtype.kind == 'c':
        if dtype.itemsize == 8:
            return c8
        elif dtype.itemsize == 16:
            return c16
        elif dtype.itemsize == 32:
            return c32
    elif dtype.kind == 'O':
        return O

    raise NotImplementedError("dtype %s not supported" % (dtype,))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
