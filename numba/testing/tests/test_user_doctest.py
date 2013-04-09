import doctest

import numba

@numba.autojit
def func(value):
    """
    >>> func(10.0)
    10.0
    """
    return value

numba.testmod()

@numba.autojit
def func2(value):
    """
    >>> raise ValueError("I am a message")
    Traceback (most recent call last):
        ...
    ValueError: I am a ...
    """

numba.testmod(verbosity=2, optionflags=doctest.ELLIPSIS)
