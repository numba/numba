"""
>>> from numba.tests.closures import test_closure_type_inference
"""

import numpy as np

from numba import *
from numba import error
from numba.testing.test_support import testmod

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
        print((a * b))

    inner()
    a = float(a)
    b = a * a # + 1j # Promotion issue
    return inner

@autojit
def test_cellvar_promotion_error(a):
    """
    >>> from numba.minivect import minierror
    >>> try:
    ...     test_cellvar_promotion_error(10)
    ... except error.UnpromotableTypeError as e:
    ...     print(sorted(e.args, key=str))
    [(int, string)]
    """
    b = int(a) * 2

    @jit(void())
    def inner():
        print((a * b))

    inner()
    a = np.empty(10, dtype=np.double)
    b = "hello"
    return inner

#test_cellvar_promotion(10)
#test_cellvar_promotion_error(10)
testmod()
