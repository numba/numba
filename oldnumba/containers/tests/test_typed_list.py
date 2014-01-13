from numba import *
import numba as nb
from numba.testing.test_support import autojit_py3doc
@autojit
def index(type):
    """
    >>> index(int_)
    ['[0, 1, 2]', '0', '1', '2']
    >>> assert index(int_) == index.py_func(int_)

    >>> index(float_)
    ['[0.0, 1.0, 2.0]', '0.0', '1.0', '2.0']
    >>> assert index(float_) == index.py_func(float_)

    >>> index(complex128)
    ['[0j, (1+0j), (2+0j)]', '0j', '(1+0j)', '(2+0j)']
    >>> assert index(complex128) == index.py_func(complex128)
    """
    tlist = nb.typedlist(type)
    tlist.append(0)
    tlist.append(1)
    tlist.append(2)
    return [str(tlist), str(tlist[0]), str(tlist[1]), str(tlist[2])]

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
    tlist = nb.typedlist(type)
    tlist.append(0)
    tlist.append(1)
    tlist.append(2)
    return tlist[4]

@autojit_py3doc
def append(type):
    """
    >>> append(int_)
    (0, 1, 2, 3)
    """
    tlist = nb.typedlist(type)
    l1 = len(tlist)
    tlist.append(0)
    l2 = len(tlist)
    tlist.append(1)
    l3 = len(tlist)
    tlist.append(2)
    l4 = len(tlist)
    return l1, l2, l3, l4

@autojit_py3doc
def append_many(type):
    """
    >>> append_many(int_)
    1000
    """
    tlist = nb.typedlist(type)
    for i in range(1000):
        tlist.append(i)
    return len(tlist)

@autojit_py3doc
def pop(type):
    """
    >>> pop(int_)
    2
    1
    0
    (3, 2, 1, 0)
    """
    tlist = nb.typedlist(type)
    for i in range(3):
        tlist.append(i)

    l1 = len(tlist)
    print((tlist.pop()))
    l2 = len(tlist)
    print((tlist.pop()))
    l3 = len(tlist)
    print((tlist.pop()))
    l4 = len(tlist)
    return l1, l2, l3, l4

@autojit_py3doc
def pop_many(type):
    """
    >>> pop_many(int_)
    (1000, 0)
    """
    tlist = nb.typedlist(type)
    for i in range(1000):
        tlist.append(i)

    initial_length = len(tlist)

    for i in range(1000):
        tlist.pop()

    return initial_length, len(tlist)

@autojit_py3doc
def from_iterable(type, iterable):
    """
    >>> from_iterable(int_, [1, 2, 3])
    [1, 2, 3]
    >>> from_iterable(int_, (1, 2, 3))
    [1, 2, 3]
    >>> from_iterable(int_, (x for x in [1, 2, 3]))
    [1, 2, 3]

    >>> from_iterable(float_, [1, 2, 3])
    [1.0, 2.0, 3.0]
    >>> from_iterable(float_, (1, 2, 3))
    [1.0, 2.0, 3.0]
    >>> from_iterable(float_, (x for x in [1, 2, 3]))
    [1.0, 2.0, 3.0]

    >>> from_iterable(int_, [1, object(), 3])
    Traceback (most recent call last):
        ...
    TypeError: an integer is required

    >>> from_iterable(int_, object())
    Traceback (most recent call last):
        ...
    TypeError: 'object' object is not iterable
    """
    return nb.typedlist(type, iterable)

@autojit_py3doc
def test_insert(type):
    """
    >>> test_insert(int_)
    [0, 1, 2, 3, 4, 5]
    """
    tlist = nb.typedlist(type, [1,3])
    tlist.insert(0,0)
    tlist.insert(2,2)
    tlist.insert(4,4)
    tlist.insert(8,5)
    return tlist

@autojit_py3doc
def test_remove(type):
    """
    >>> test_remove(int_)
    4
    3
    2
    [1, 3]
    """
    tlist = nb.typedlist(type, range(5))
    tlist.remove(0)
    print (len(tlist))
    tlist.remove(2)
    print (len(tlist))
    tlist.remove(4)
    print (len(tlist))
    return tlist

@autojit_py3doc
def test_count(type, L):
    """
    >>> test_count(int_, [1, 2, 3, 4, 5, 1, 2])
    (0, 1, 2)
    """
    tlist = nb.typedlist(type, L)
    return tlist.count(0), tlist.count(3), tlist.count(1)

@autojit_py3doc
def test_count_complex(type, L):
    """
    >>> test_count_complex(complex128, [1+1j, 1+2j, 2+1j, 2+2j, 1+1j, 2+2j, 2+2j])
    (1, 2, 3)
    """
    tlist = nb.typedlist(type, L)
    return tlist.count(1+2j), tlist.count(1+1j), tlist.count(2+2j)

@autojit_py3doc
def test_index(type):
    """
    >>> test_index(int_)
    (0, 2, 4)
    """
    tlist = nb.typedlist(type, [5, 4, 3, 2, 1])
    return tlist.index(5), tlist.index(3), tlist.index(1)

@autojit
def test_reverse(type, value):
    """
    >>> test_reverse(int_, range(10))
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    >>> test_reverse(int_, range(11))
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    >>> test_reverse(float_, range(10))
    [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    >>> test_reverse(float_, range(11))
    [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    """
    tlist = nb.typedlist(type, value)
    tlist.reverse()
    return tlist

#@autojit
#def test_sort(type, value):
#    """
#    >>> test_sort(int_, range(5, 10) + range(5) + range(10, 15))
#    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#    """
#    tlist = nb.typedlist(type, value)
#    tlist.sort()
#    return tlist

def test(module):
    nb.testing.testmod(module)

if __name__ == "__main__":
    import __main__ as module
else:
    import test_typed_list as module

test(module)
__test__ = {}
