import ctypes.util
import warnings

import numba
# from numba.typesystem.defaults import numba_typesystem as ts
from numba.typesystem import numbatypes as ts
from numba.typesystem.ctypestypes import ctypes_map
import numba.utils

#-------------------------------------------------------------------
# CTypes Types for Type Checking
#-------------------------------------------------------------------

_ctypes_scalar_type = type(ctypes.c_int)
_ctypes_func_type = type(ctypes.CFUNCTYPE(ctypes.c_int))
_ctypes_pointer_type = type(ctypes.POINTER(ctypes.c_int))
_ctypes_array_type = type(ctypes.c_int * 2)

CData = type(ctypes.c_int(10)).__mro__[-2]

#-------------------------------------------------------------------
# Check Whether values are ctypes values
#-------------------------------------------------------------------

def is_ctypes_function(value):
    return isinstance(type(value), _ctypes_func_type)

def is_ctypes_value(ctypes_value):
    return isinstance(ctypes_value, CData)

def is_ctypes_struct_type(ctypes_type):
    return (isinstance(ctypes_type, type) and
            issubclass(ctypes_type, ctypes.Structure))

def is_ctypes_type(ctypes_type):
    return (
       (isinstance(ctypes_type, _ctypes_scalar_type)) or
       is_ctypes_struct_type(ctypes_type)
    )

def is_ctypes(value):
    "Check whether the given value is a ctypes value"
    return is_ctypes_value(value) or is_ctypes_type(value)

#-------------------------------------------------------------------
# Type mapping (ctypes -> numba)
#-------------------------------------------------------------------

def from_ctypes_type(ctypes_type):
    """
    Convert a ctypes type to a numba type
    """
    if numba.utils.hashable(ctypes_type) and ctypes_type in ctypes_map:
        return ctypes_map[ctypes_type]
    elif ctypes_type is ctypes.c_void_p:
        return from_ctypes_type(None).pointer()
    elif isinstance(ctypes_type, _ctypes_pointer_type):
        return from_ctypes_type(ctypes_type._type_).pointer()
    elif isinstance(ctypes_type, _ctypes_array_type):
        base_type = from_ctypes_type(ctypes_type._type_)
        return ts.carray(base_type, ctypes_type._length_)
    elif issubclass(ctypes_type, ctypes.Structure):
        fields = [(name, from_ctypes_type(field_type))
                      for name, field_type in ctypes_type._fields_]
        return ts.struct_(fields)
    else:
        raise NotImplementedError(ctypes_type)

def from_ctypes_value(value):
    """
    Convert a ctypes value to a numba type
    """
    if is_ctypes_type(value):
        # Value is a ctypes type, e.g. c_int
        return ts.meta(from_ctypes_type(value))

    elif is_ctypes_function(value):
        # TODO: move this to from_ctypes_type
        if value.argtypes is None:
            warnings.warn(
                "ctypes function %s has no argument types set" % (value,))
            return ts.object_

        restype = from_ctypes_type(value.restype)
        argtypes = [from_ctypes_type(at) for at in value.argtypes]
        signature = ts.function(return_type=restype, args=argtypes)
        return signature

    elif is_ctypes_type(type(value)) or hasattr(value, '_type_'):
        # Value is a ctypes value, e.g. c_int(10)
        result_type = from_ctypes_type(type(value))

        if result_type.is_pointer:
            # Handle ctypes pointers
            try:
                ctypes.cast(value, ctypes.c_void_p)
            except ctypes.ArgumentError:
                pass
            else:
                addr_int = ctypes.cast(value, ctypes.c_void_p).value
                result_type = ts.known_pointer(result_type.base_type, addr_int)

        return result_type

    else:
        raise NotImplementedError(value)
