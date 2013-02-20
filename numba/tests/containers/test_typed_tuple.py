from numba import *
import numba as nb

@autojit
def index(type):
    """
    >>> index(int_)
    ['(0, 1, 2)', '0', '1', '2']
    >>> assert index(int_) == index.py_func(int_)

    >>> index(float_)
    ['(0.0, 1.0, 2.0)', '0.0', '1.0', '2.0']
    >>> assert index(float_) == index.py_func(float_)

    >>> index(complex128)
    ['(0j, (1+0j), (2+0j))', '0j', '(1+0j)', '(2+0j)']
    >>> assert index(complex128) == index.py_func(complex128)
    """
    ttuple = nb.typedtuple(type, (0, 1, 2))
    return map(str, [ttuple, ttuple[0], ttuple[1], ttuple[2]])

@autojit
def index_error(type):
    """
    >>> index_error(int_)
    Traceback (most recent call last):
        ...
    IndexError: list index out of range

    >>> index_error(float_)
    Traceback (most recent call last):
        ...
    IndexError: list index out of range
    """
    ttuple = nb.typedtuple(type, (0, 1, 2))
    return ttuple[4]

@autojit
def from_iterable(type, iterable):
    """
    >>> from_iterable(int_, [1, 2, 3])
    (1, 2, 3)
    >>> from_iterable(int_, (1, 2, 3))
    (1, 2, 3)
    >>> from_iterable(int_, (x for x in [1, 2, 3]))
    (1, 2, 3)

    >>> from_iterable(float_, [1, 2, 3])
    (1.0, 2.0, 3.0)
    >>> from_iterable(float_, (1, 2, 3))
    (1.0, 2.0, 3.0)
    >>> from_iterable(float_, (x for x in [1, 2, 3]))
    (1.0, 2.0, 3.0)

    >>> from_iterable(int_, [1, object(), 3])
    Traceback (most recent call last):
        ...
    TypeError: an integer is required

    >>> from_iterable(int_, object())
    Traceback (most recent call last):
        ...
    TypeError: 'object' object is not iterable
    """
    return nb.typedtuple(type, iterable)

@autojit
def test_count(type):
    """
    >>> test_count(int_)
    Traceback (most recent call last):
        ...
    NotImplementedError: 'count' method
    """
    ttuple = nb.typedtuple(type, [1, 2, 3, 4, 5, 1, 2])
    return ttuple.count(0), ttuple.count(3), ttuple.count(1)

def test():
    nb.testmod()

if __name__ == "__main__":
    test()
