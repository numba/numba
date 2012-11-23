# Adapted from cython/tests/run/for_in_range.pyx

from numba.tests.test_support import *

@autojit
def test_modify():
    """
    >>> test_modify()
    Warning 23:11: local variable 'i' might be referenced before assignment
    0
    1
    2
    3
    4
    <BLANKLINE>
    (4L, 0L)
    """
    n = 5
    for i in range(n):
        print i
        n = 0
    print
    return i,n

@autojit
def test_negindex():
    """
    >>> test_negindex()
    Warning 41:11: local variable 'i' might be referenced before assignment
    6
    5
    4
    3
    2
    (2L, 0L)
    """
    n = 5
    for i in range(n+1, 1, -1):
        print i
        n = 0
    return i,n

@autojit
def test_negindex_inferred():
    """
    >>> test_negindex_inferred()
    Warning 58:11: local variable 'i' might be referenced before assignment
    5
    4
    3
    2
    (2L, 0L)
    """
    n = 5
    for i in range(n, 1, -1):
        print i
        n = 0
    return i,n

@autojit
def test_fix():
    """
    >>> test_fix()
    Warning 76:11: local variable 'i' might be referenced before assignment
    0
    1
    2
    3
    4
    <BLANKLINE>
    4L
    """
    for i in range(5):
        print i
    print
    return i

@autojit
def test_break():
    """
    >>> test_break()
    Warning 98:11: local variable 'i' might be referenced before assignment
    0
    1
    2
    <BLANKLINE>
    (2L, 0L)
    """
    n = 5
    for i in range(n):
        print i
        n = 0
        if i == 2:
            break
    else:
        print "FAILED!"
    print
    return i, n

@autojit
def test_return():
    """
    >>> test_return()
    0
    1
    2
    (2L, 0L)
    """
    n = 5
    for i in range(n):
        print i
        n = 0
        if i == 2:
            return i,n
    print
    return "FAILED!"

#print test_negindex()
testmod()