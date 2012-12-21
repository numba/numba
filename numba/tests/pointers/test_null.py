import ctypes
import numba
from numba import *

#intp = ctypes.POINTER(ctypes.c_int)
#voidp = ctypes.c_void_p

intp = int_.pointer()
voidp = void.pointer()

@autojit
def test_compare_null():
    """
    >>> test_compare_null()
    True
    """
    return intp(Py_uintptr_t(0)) == NULL

@autojit
def test_compare_null_attribute():
    """
    >>> test_compare_null_attribute()
    True
    """
    return voidp(Py_uintptr_t(0)) == numba.NULL

if __name__ == '__main__':
#    test_compare_null()
#    test_compare_null_attribute()
    numba.testmod()