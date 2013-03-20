import sys

import numba

from numba.tests.test_support import autojit_py3doc
# NOTE: See also issues/test_issue_56

@autojit_py3doc
def test_bitwise_and(a, b):
    """
    >>> test_bitwise_and(0b01, 0b10)
    0L
    >>> test_bitwise_and(0b01, 0b11)
    1L

    >>> test_bitwise_and(0b01, 2.0)
    Traceback (most recent call last):
        ...
    NumbaError: 27:15: Expected an int, or object, or bool

    >>> test_bitwise_and(2.0, 0b01)
    Traceback (most recent call last):
        ...
    NumbaError: 27:11: Expected an int, or object, or bool

    """
    return a & b

@autojit_py3doc
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
    NumbaError: 54:15: Expected an int, or object, or bool

    >>> test_bitwise_or(2.0, 0b01)
    Traceback (most recent call last):
        ...
    NumbaError: 54:11: Expected an int, or object, or bool

    """
    return a | b


@autojit_py3doc
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
    NumbaError: 82:15: Expected an int, or object, or bool

    >>> test_bitwise_xor(2.0, 0b01)
    Traceback (most recent call last):
        ...
    NumbaError: 82:11: Expected an int, or object, or bool

    """
    return a ^ b

@autojit_py3doc
def test_shift_left(a, b):
    """
    >>> test_shift_left(5, 2)
    20L
    >>> test_shift_left(-5, 2)
    -20L
    """
    return a << b

@autojit_py3doc
def test_shift_right(a, b):
    """
    >>> test_shift_right(20, 2)
    5L
    >>> test_shift_right(-20, 2)
    -5L
    """
    return a >> b

@autojit_py3doc
def test_invert(a):
    """
    >>> test_invert(5)
    -6L
    >>> test_invert(-5)
    4L
    """
    return ~a


numba.testmod()
