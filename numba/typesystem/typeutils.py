from numba.typesystem import *

def index_type(type):
    "Result of indexing a value of the given type with an integer index"
    if type.is_array:
        result = type.copy()
        result.ndim -= 1
        if result.ndim == 0:
            result = result.dtype
    else:
        result = type.base_type

    return result