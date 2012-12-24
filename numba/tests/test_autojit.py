from numba import *

#from numba import testmod
from numba.tests.test_support import testmod

@autojit(nopython=True)
def autojit_arg(result):
    return result + 1

@jit(float_(float_))
def jit_arg(result):
    return result + 1

@autojit(nopython=True)
def autojit_as_arg(autojit_arg, value):
    """
    >>> autojit_as_arg(autojit_arg, 0.0)
    10.0
    """
    result = value
    for i in range(10):
        result = autojit_arg(result)
    return result

@autojit(nopython=True)
def jit_as_arg(jit_arg, value):
    """
    >>> jit_as_arg(jit_arg, 0.0)
    10.0
    """
    result = value
    for i in range(10):
        result = jit_arg(result)
    return result


testmod()