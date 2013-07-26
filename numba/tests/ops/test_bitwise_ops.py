import sys
import numba
from numba.testing.test_support import autojit_py3doc
# NOTE: See also issues/test_issue_56

autojit_py3doc = autojit_py3doc(warn=False, warnstyle='simple')

@autojit_py3doc
def test_bitwise_and(a, b):
    """
    >>> test_bitwise_and(0b01, 0b10)
    0
    >>> test_bitwise_and(0b01, 0b11)
    1

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
    0
    >>> test_bitwise_or(0b00, 0b01)
    1
    >>> test_bitwise_or(0b10, 0b00)
    2
    >>> test_bitwise_or(0b01, 0b10)
    3
    >>> test_bitwise_or(0b01, 0b11)
    3

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
    0
    >>> test_bitwise_xor(0b00, 0b01)
    1
    >>> test_bitwise_xor(0b10, 0b00)
    2
    >>> test_bitwise_xor(0b01, 0b10)
    3
    >>> test_bitwise_xor(0b01, 0b11)
    2

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
    20
    >>> test_shift_left(-5, 2)
    -20
    """
    return a << b

@autojit_py3doc
def test_shift_right(a, b):
    """
    >>> test_shift_right(20, 2)
    5
    >>> test_shift_right(-20, 2)
    -5
    """
    return a >> b

@autojit_py3doc
def test_invert(a):
    """
    >>> test_invert(5)
    -6
    >>> test_invert(-5)
    4
    """
    return ~a


numba.testing.testmod()
