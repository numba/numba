import sys

from numba import extension_types
from numba import *

def format_str(msg, *values):
    return msg % values

@jit
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

    @double()
    def getvalue(self):
        "Return value"
        return self.value

    @void(double)
    def setvalue(self, value):
        "Set value"
        self.value = value

    @object_()
    def __repr__(self):
        return format_str('MyExtension%s', self.value)

@jit
class ObjectAttrExtension(object):
    """
    >>> obj = ObjectAttrExtension(10.0, 'blah')
    Traceback (most recent call last):
        ...
    TypeError: a float is required
    >>> obj = ObjectAttrExtension(10.0, 3.5)
    >>> obj.value1
    10.0
    >>> obj = ObjectAttrExtension('hello', 9.3)
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
    >>> obj.method()
    MyExtension10.0

    >>> obj.method2(15.0)
    30.0

    # This leads to a segfault, why?
#    >>> obj._numba_attrs._fields_
#    [('value2', <class 'ctypes.c_double'>), ('value1', <class 'ctypes.py_object'>)]
    """

    def __init__(self, value1, value2):
        self.value1 = object_(value1)
        self.value2 = double(value2)

    @object_()
    def getvalue(self):
        "Return value"
        return self.value1

    @void(double)
    def setvalue(self, value):
        "Set value"
        self.value1 = value

    @object_()
    def method(self):
        return self.getvalue()

    @object_(int32)
    def method2(self, new_value):
        self.setvalue(new_value * 2)
        result = self.method()
        return result

exttype = ObjectAttrExtension.exttype

@jit
class ExtensionTypeAsAttribute(object):
    """
    >>> print ExtensionTypeAsAttribute.exttype
    <Extension ExtensionTypeAsAttribute({'attr': <Extension ObjectAttrExtension>})>
    """

    def __init__(self, attr):
        self.attr = exttype(attr)


if __name__ == '__main__':
#    print ExtensionTypeAsAttribute.exttype
    import doctest
    doctest.testmod()