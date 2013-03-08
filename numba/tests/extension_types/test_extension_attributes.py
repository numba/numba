"""
>>> Base(10, 11.0).value1
10.0
>>> Base(10, 11.0).value2
11


>>> obj = Derived(10, 11.0)
>>> obj.setvalue(9)
>>> obj.value3
9.0
"""

import sys
import numba
from numba import *
from nose.tools import raises

@jit
class Base(object):

    value1 = double
    value2 = int_

    @void(int_, double)
    def __init__(self, value1, value2):
        self.value1 = value1
        self.value2 = value2

@jit
class Derived(Base):

    value3 = float_

    @void(int_)
    def setvalue(self, value):
        self.value3 = value

if __name__ == '__main__':
    import numba
    numba.testmod()
