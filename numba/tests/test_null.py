import ctypes
from numba import *

#intp = ctypes.POINTER(ctypes.c_int)
#voidp = ctypes.c_void_p

intp = int_.pointer()
voidp = void.pointer()

@autojit
def test_compare_null():
    """
    >>> test_compare_null()
    """
    return intp(Py_uintptr_t(0)) == NULL

@autojit
def test_compare_null_attribute():
    """
    >>> test_compare_null()
    """
    return voidp(0) == numba.NULL

if __name__ == '__main__':
    test_compare_null()
#    import doctest
#    doctest.testmod()
