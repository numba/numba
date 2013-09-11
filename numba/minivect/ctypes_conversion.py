# -*- coding: utf-8 -*-
"""
Convert a minivect type to a ctypes type and an llvm function to a
ctypes function.
"""
from __future__ import print_function, division, absolute_import

import math
import warnings

from .miniutils import ctypes
from .minitypes import *

try:
    from ctypes import *
except ImportError:
    pass

def convert_to_ctypes(type):
    """
    Convert the minitype to a ctypes type

    >>> from minitypes import *
    >>> assert convert_to_ctypes(int32) == ctypes.c_int32
    >>> assert convert_to_ctypes(int64) == ctypes.c_int64
    >>> assert convert_to_ctypes(uint32) == ctypes.c_uint32
    >>> assert convert_to_ctypes(uint64) == ctypes.c_uint64
    >>> assert convert_to_ctypes(short) == ctypes.c_short
    >>> assert convert_to_ctypes(int_) == ctypes.c_int
    >>> assert convert_to_ctypes(long_) == ctypes.c_long
    >>> assert convert_to_ctypes(float_) == ctypes.c_float
    >>> assert convert_to_ctypes(double) == ctypes.c_double
    >>> #convert_to_ctypes(complex64)
    >>> #convert_to_ctypes(complex128)
    >>> #convert_to_ctypes(complex256)
    """
    from . import minitypes

    if type.is_pointer:
        return ctypes.POINTER(convert_to_ctypes(type.base_type))
    elif type.is_object or type.is_array:
        return ctypes.py_object
    elif type.is_float:
        if type.itemsize == 4:
            return ctypes.c_float
        elif type.itemsize == 8:
            return ctypes.c_double
        else:
            return ctypes.c_longdouble
    elif type.is_numpy_intp or type.is_py_ssize_t:
        if minitypes._plat_bits == 32:
            return ctypes.c_int32
        else:
            return ctypes.c_int64
    elif type == minitypes.int_:
        return ctypes.c_int
    elif type == minitypes.uint:
        return ctypes.c_uint
    elif type == minitypes.long_:
        return ctypes.c_long
    elif type == minitypes.ulong:
        return ctypes.c_ulong
    # TODO: short, long long, etc
    elif type.is_int:
        item_idx = int(math.log(type.itemsize, 2))
        if type.signed:
            values = [ctypes.c_int8, ctypes.c_int16, ctypes.c_int32,
                      ctypes.c_int64]
        else:
            values = [ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32,
                      ctypes.c_uint64]
        return values[item_idx]
    elif type.is_complex:
        from . import complex_support
        if type.itemsize == 8:
            return complex_support.Complex64
        elif type.itemsize == 16:
            return complex_support.Complex128
        else:
            return complex_support.Complex256
    elif type.is_c_string:
        return ctypes.c_char_p
    elif type.is_function:
        return_type = convert_to_ctypes(type.return_type)
        arg_types = tuple(convert_to_ctypes(arg_type)
                              for arg_type in type.args)
        return ctypes.CFUNCTYPE(return_type, *arg_types)
    elif type.is_void:
        return None
    elif type.is_carray:
        return convert_to_ctypes(type.base_type) * type.size
    elif type.is_struct:
        class Struct(ctypes.Structure):
            _fields_ = [(field_name, convert_to_ctypes(field_type))
                            for field_name, field_type in type.fields]
        return Struct
    else:
        raise NotImplementedError(type)


def get_ctypes_func(func, llvm_func, llvm_execution_engine, context):
    "Get a ctypes function from an llvm function"
    ctypes_func_type = convert_to_ctypes(func.type)
    p = llvm_execution_engine.get_pointer_to_function(llvm_func)
    return ctypes_func_type(p)

def get_data_pointer(numpy_array, array_type):
    "Get a ctypes typed data pointer for the numpy array with type array_type"
    dtype_pointer = array_type.dtype.pointer()
    return numpy_array.ctypes.data_as(convert_to_ctypes(dtype_pointer))

def get_pointer(context, llvm_func):
    "Get a pointer to the LLVM function (int)"
    from numba.codegen.llvmcontext import LLVMContextManager
    return LLVMContextManager().execution_engine.get_pointer_to_function(llvm_func)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
