"""
>>> test_refcounting()
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

>>> sys.getrefcount(object())
1
>>> sys.getrefcount(fresh_obj())
1
>>> sys.getrefcount(fresh_obj2())
1
>>> sys.getrefcount(fresh_obj3())
1
>>> sys.getrefcount(fresh_obj4())
1
>>> sys.getrefcount(fresh_obj5())
1
>>> sys.getrefcount(fresh_obj6())
1

Test list/dict/tuple literals

>>> sys.getrefcount(fresh_obj7())
1
>>> sys.getrefcount(fresh_obj7()[0])
1
>>> sys.getrefcount(fresh_obj8())
1
>>> sys.getrefcount(fresh_obj8()["value"])
1
>>> sys.getrefcount(fresh_obj9())
1
>>> sys.getrefcount(fresh_obj9()[0])
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
... except TypeError, e:
...     del e
...     sys.exc_clear()
... else:
...     raise Exception("An exception should have been raised")
...
>>> sys.getrefcount(obj)
2

>>> obj1, obj2 = object(), np.arange(10)
>>> sys.getrefcount(obj1), sys.getrefcount(obj2)
(2, 2)
>>> x, y = count_arguments(obj1, obj2)
>>> assert x is y is obj2
>>> sys.getrefcount(x)
4

>>> def test_count_arguments(f, obj):
...     print sys.getrefcount(obj)
...     f(obj)
...     print sys.getrefcount(obj)
...
>>> test_count_arguments(count_arguments2, object())
3
3
>>> test_count_arguments(count_arguments2, np.arange(10))
3
3
>>> test_count_arguments(count_arguments3, object())
3
3
>>> test_count_arguments(count_arguments3, np.arange(10))
3
3
"""

import sys
import ctypes

from numba import *
import numpy as np

from numba.tests import test_support

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
    L = np.array([Unique(i) for i in range(10)], dtype=np.object)
    assert all(sys.getrefcount(obj) == 3 for obj in L)
    with test_support.StdoutReplacer() as out:
        use_objects(L)
    # print out.getvalue()

    # This fails in nose
    #expected = "\n".join("Unique(%d)" % i for i in range(10)) + '\n'
    #print out.getvalue() == expected
    print [sys.getrefcount(obj) for obj in L]

@autojit(backend='ast', warn=False)
def fresh_obj():
    x = object()
    return x

@autojit(backend='ast', warn=False)
def fresh_obj2():
    return object()

@autojit(backend='ast', warn=False)
def fresh_obj3():
    x = object()
    y = x
    return y

@autojit(backend='ast', warn=False)
def fresh_obj4():
    x = np.ones(1, dtype=np.double)
    y = x
    return y

@autojit(backend='ast', warn=False)
def fresh_obj5():
    return np.ones(1, dtype=np.double)

@autojit(backend='ast', warn=False)
def fresh_obj6():
    x = np.ones(1, dtype=np.double)
    y = x
    return x

@autojit(backend='ast', warn=False)
def fresh_obj7():
    x = np.ones(1, dtype=np.double)
    return [x]

@autojit(backend='ast', warn=False)
def fresh_obj8():
    x = np.ones(1, dtype=np.double)
    return {"value": x}

@autojit(backend='ast', warn=False)
def fresh_obj9():
    x = np.ones(1, dtype=np.double)
    return (x,)

@autojit(backend='ast', warn=False)
def index_count(L):
    x = L[0]
    return x

@autojit(backend='ast', warn=False)
def attr_count(obj):
    x = obj.value
    return x

@autojit(backend='ast', warn=False)
def exc(obj):
    x = obj
    return object()('boom')

@autojit(backend='ast', warn=False)
def count_arguments(x, y):
    x = y
    y = x
    a = x
    b = y
    return x, y

@autojit(backend='ast', warn=False)
def count_arguments2(obj):
    pass

@autojit(backend='ast', warn=False)
def count_arguments3(obj):
    x = obj

if __name__ == "__main__":
#    print sys.getrefcount(fresh_obj())
#    exc(object())
    import doctest
    doctest.testmod()
