

import sys

from numba import extension_types
from numba import *

def format_str(msg, *values):
    return msg % values

@autojit
class MyExtension(object):
    """
    >>> obj = MyExtension(10.0)
    >>> obj.value
    10.0
    >>> obj._numba_attrs.value
    10.0
    >>> obj.setvalue(20.0)
    >>> obj.getvalue()
    20.0
    >>> obj.value
    20.0
    >>> obj.getvalue.__name__
    'getvalue'
    >>> obj.getvalue.__doc__
    'Return value'
    >>> type(obj.getvalue.im_func)
    <type 'cython_function_or_method'>
    >>> obj._numba_attrs._fields_
    [('value', <class 'ctypes.c_double'>)]
    """

    @void(double)
    def __init__(self, myfloat):
        self.value = myfloat

    def getvalue(self):
        "Return value"
        return self.value

    @void(double)
    def setvalue(self, value):
        "Set value"
        self.value = value

    def __repr__(self):
        return format_str('MyExtension%s', self.value)

@autojit
class ObjectAttrExtension(object):
    """
    >>> obj = ObjectAttrExtension(10.0, 'blah')
    >>> obj.value1
    10.0
    >>> obj = ObjectAttrExtension('hello', 'world')
    >>> obj.value1
    'hello'
    >>> obj.setvalue(20.0)
    >>> obj.getvalue()
    20.0
    >>> obj.value1 = MyExtension(10.0)
    >>> obj.value1
    MyExtension10.0
    >>> obj.getvalue()
    MyExtension10.0
    >>> obj._numba_attrs._fields_
    [('value2', <class 'ctypes.c_double'>), ('value1', <class 'ctypes.py_object'>)]
    """

    def __init__(self, value1, value2):
        self.value1 = object_(value1)
        self.value2 = double(value2)

    def getvalue(self):
        "Return value"
        return self.value1

    @void(double)
    def setvalue(self, value):
        "Set value"
        self.value1 = value

if __name__ == '__main__':
    import doctest
    doctest.testmod()