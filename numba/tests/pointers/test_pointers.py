import ctypes
import numba
from numba import *

int32p = int32.pointer()
voidp = void.pointer()

@autojit
def test_pointer_arithmetic():
    """
    >>> test_pointer_arithmetic()
    48
    """
    p = int32p(Py_uintptr_t(0))
    p = p + 10
    p += 2
    return Py_uintptr_t(p) # 0 + 4 * 12


if __name__ == '__main__':
#    print test_pointer_arithmetic()

    import doctest
    doctest.testmod()
