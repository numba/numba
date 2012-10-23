

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
    >>> obj = ObjectAttrExtension(10.0)
    >>> obj.value
    10.0
    >>> obj = ObjectAttrExtension('hello')
    >>> obj.value
    'hello'
    >>> obj.setvalue(20.0)
    >>> obj.getvalue()
    20.0
    >>> obj.value = MyExtension(10.0)
    >>> obj.value
    MyExtension10.0
    """

    @void(object_)
    def __init__(self, myfloat):
        self.value = myfloat

    def getvalue(self):
        "Return value"
        return self.value

    @void(double)
    def setvalue(self, value):
        "Set value"
        self.value = value

if __name__ == '__main__':
    import doctest
    doctest.testmod()