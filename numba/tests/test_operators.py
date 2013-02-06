import sys

import numba
from numba import *



@autojit
def test_bitwise_and(a, b):
    """
    >>> test_bitwise_and(0b01, 0b10)
    0L
    >>> test_bitwise_and(0b01, 0b11)
    1L

    >>> test_bitwise_and(0b01, 2.0)
    Traceback (most recent call last):
        ...
    NumbaError: 27:15: Expected an int

    >>> test_bitwise_and(2.0, 0b01)
    Traceback (most recent call last):
        ...
    NumbaError: 27:11: Expected an int

    """
    return a & b

@autojit
def test_bitwise_or(a, b):
    """
    >>> test_bitwise_or(0b00, 0b00)
    0L
    >>> test_bitwise_or(0b00, 0b01)
    1L
    >>> test_bitwise_or(0b10, 0b00)
    2L
    >>> test_bitwise_or(0b01, 0b10)
    3L
    >>> test_bitwise_or(0b01, 0b11)
    3L

    >>> test_bitwise_or(0b01, 2.0)
    Traceback (most recent call last):
        ...
    NumbaError: 54:15: Expected an int

    >>> test_bitwise_or(2.0, 0b01)
    Traceback (most recent call last):
        ...
    NumbaError: 54:11: Expected an int

    """
    return a | b


@autojit
def test_bitwise_xor(a, b):
    """
    >>> test_bitwise_xor(0b00, 0b00)
    0L
    >>> test_bitwise_xor(0b00, 0b01)
    1L
    >>> test_bitwise_xor(0b10, 0b00)
    2L
    >>> test_bitwise_xor(0b01, 0b10)
    3L
    >>> test_bitwise_xor(0b01, 0b11)
    2L

    >>> test_bitwise_xor(0b01, 2.0)
    Traceback (most recent call last):
        ...
    NumbaError: 82:15: Expected an int

    >>> test_bitwise_xor(2.0, 0b01)
    Traceback (most recent call last):
        ...
    NumbaError: 82:11: Expected an int

    """
    return a ^ b

numba.testmod()