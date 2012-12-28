import numpy as np
import numba
from numba import *

@jit
class Foo(object):
    def __init__(self, arg):
        self.arg = double(arg)

def test_typeof_pure(arg):
    """
    >>> test_typeof_pure(10)
    int
    >>> test_typeof_pure(10.0)
    double
    >>> print test_typeof_pure(Foo(10))
    <Extension Foo({'arg': double})>
    """
    return numba.typeof(arg)

@autojit
def test_typeof_numba(a, b):
    """
    >>> test_typeof_numba(10, 11.0)
    21L
    >>> test_typeof_numba(11.0, 10)
    21.0
    >>> test_typeof_numba(np.arange(10), 1)
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
    """
    return numba.typeof(a)(a + b)

@autojit
def test_typeof_numba2(arg):
    """
    >>> test_typeof_numba2(10)
    (10+0j)
    """
    x = 1 + 2j
    arg = numba.typeof(x)(arg)
    return numba.typeof(arg)(arg)

@autojit
def test_typeof_numba3(arg):
    """
    >>> print test_typeof_numba3(10)
    int
    >>> print test_typeof_numba3(Foo(10))
    <Extension Foo({'arg': double})>
    """
    return numba.typeof(arg)

numba.testmod()