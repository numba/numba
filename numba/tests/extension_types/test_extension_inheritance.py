"""
Test Python-level inheritance

>>> Base(10.0)
Base(10.0)
>>> Derived(20.0)
Derived(20.0)

>>> Base(10.0).py_method()
10.0
>>> Derived(10.0).py_method()
10.0
>>> Base.py_method(object())
Traceback (most recent call last):
    ...
TypeError: unbound method cython_function_or_method object must be called with Base instance as first argument (got object instance instead)

Test numba virtual methods

>>> Base(4.0).method()
4.0
>>> Base(4.0).getvalue()
4.0

>>> Derived(4.0).method()
8.0
>>> Derived(4.0).getvalue()
8.0
>>> obj = Derived(4.0)
>>> obj.value2 = 3.0
>>> obj.method()
12.0
"""

import sys

from numba import *

def format_str(msg, *values):
    return msg % values

@jit
class Base(object):

    @void(double)
    def __init__(self, value):
        self.value = value

    @double()
    def getvalue(self):
        "Return value"
        return self.value

    @void(double)
    def setvalue(self, value):
        "Set value"
        self.value = value

    @double()
    def method(self):
        return self.getvalue()

    @double()
    def py_method(self):
        return self.value

    @object_()
    def __repr__(self):
        return format_str('Base(%s)', self.value)

@jit
class Derived(Base):

    @void(double)
    def __init__(self, value):
        self.value = value
        self.value2 = 2.0

    @double()
    def getvalue(self):
        "Return value"
        return self.value * self.value2

    @void(double)
    def setvalue2(self, value2):
        "Set value"
        self.value2 = value2

    @object_()
    def __repr__(self):
        return format_str('Derived(%s)', self.value)

if __name__ == '__main__':
    import doctest
    doctest.testmod()