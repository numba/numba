import numpy as np

from numba import *
from numba.tests.test_support import *

@autojit
def test_cellvar_promotion(a):
    """
    >>> inner = test_cellvar_promotion(10)
    200.0
    >>> inner.__name__
    'inner'
    >>> inner()
    1000.0
    """
    b = int(a) * 2

    @jit(void())
    def inner():
        print a * b

    inner()
    a = float(a)
    b = a * a # + 1j # Promotion issue
    return inner

testmod()