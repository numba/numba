"""
>>> test_refcounting()
True
True
>>> sys.getrefcount(object())
1
>>> sys.getrefcount(fresh_obj())
1
>>> sys.getrefcount(fresh_obj2())
1
>>> sys.getrefcount(fresh_obj3())
1
>>> sys.getrefcount(index_count([object()]))
1
>>> class C(object):
...     def __init__(self, value):
...         self.value = value
...     def __del__(self):
...         print 'deleting...'
...
>>> sys.getrefcount(attr_count(C(object())))
deleting...
1

>>> obj = object()
>>> sys.getrefcount(obj)
2
>>> try:
...     exc(obj)
... except ValueError, e:
...     del e
...     sys.exc_clear()
...
>>> sys.getrefcount(obj)
2

>>> obj1, obj2 = object(), object()
>>> sys.getrefcount(obj1), sys.getrefcount(obj2)
(2, 2)
>>> x, y = count_arguments(obj1, obj2)
>>> assert x is y is obj2
>>> sys.getrefcount(x)
4
"""

import sys
import ctypes

from numba import *
import numpy as np

class Unique(object):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Unique(%d)" % self.value

@autojit(backend='ast')
def use_objects(obj_array):
    for i in range(10):
        var = obj_array[i]
        print var

def test_refcounting():
    import test_support

    L = np.array([Unique(i) for i in range(10)], dtype=np.object)
    assert all(sys.getrefcount(obj) == 3 for obj in L)
    with test_support.StdoutReplacer() as out:
        use_objects(L)

    expected = "\n".join("Unique(%d)" % i for i in range(10)) + '\n'
    print out.getvalue() == expected
    print all(sys.getrefcount(obj) == 3 for obj in L)

@autojit(backend='ast')
def fresh_obj():
    x = object()
    return x

@autojit(backend='ast')
def fresh_obj2():
    return object()

@autojit(backend='ast')
def fresh_obj3():
    x = object()
    y = x
    return y

@autojit(backend='ast')
def index_count(L):
    x = L[0]
    return x

@autojit(backend='ast')
def attr_count(obj):
    x = obj.value
    return x

@autojit(backend='ast')
def exc(obj):
    x = obj
    return int('boom')

@autojit(backend='ast')
def count_arguments(x, y):
    x = y
    y = x
    a = x
    b = y
    return x, y

if __name__ == "__main__":
    import doctest
    doctest.testmod()