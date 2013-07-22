from numba import autojit
from numba.testing.test_support import autojit_py3doc

@autojit_py3doc
def max1(x):
    """
    >>> max1([100])
    100
    >>> max1([1,2.0,3])
    3
    >>> max1([-1,-2,-3.0])
    -1
    >>> max1(1)
    Traceback (most recent call last):
        ...
    TypeError: 'int' object is not iterable
    """
    return max(x)

@autojit_py3doc
def min1(x):
    """
    >>> min1([100])
    100
    >>> min1([1,2,3.0])
    1
    >>> min1([-1,-2.0,-3])
    -3
    >>> min1(1)
    Traceback (most recent call last):
        ...
    TypeError: 'int' object is not iterable
    """
    return min(x)

@autojit_py3doc
def max2(x, y):
    """
    >>> max2(1, 2)
    2
    >>> max2(1, -2)
    1
    >>> max2(10, 10.25)
    10.25
    >>> max2(10, 9.9)
    10.0
    >>> max2(0.1, 0.25)
    0.25
    >>> max2(1, 'a')
    Traceback (most recent call last):
        ...
    UnpromotableTypeError: Cannot promote types int and string
    """
    return max(x, y)

@autojit_py3doc
def min2(x, y):
    """
    >>> min2(1, 2)
    1
    >>> min2(1, -2)
    -2
    >>> min2(10, 10.1)
    10.0
    >>> min2(10, 9.75)
    9.75
    >>> min2(0.25, 0.3)
    0.25
    >>> min2(1, 'a')
    Traceback (most recent call last):
        ...
    UnpromotableTypeError: Cannot promote types int and string
    """
    return min(x, y)

@autojit
def max4(x):
    """
    >>> max4(20)
    20.0
    """
    return max(1, 2.0, x, 14)

@autojit
def min4(x):
    """
    >>> min4(-2)
    -2.0
    """
    return min(1, 2.0, x, 14)


if __name__ == '__main__':
    import numba
    numba.testing.testmod()
