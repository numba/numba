"""
>>> test_typeof()
"""

import numba
from numba import *

def make_base(compiler):
    @compiler
    class Base(object):

        value1 = double
        value2 = int_

        @void(int_, double)
        def __init__(self, value1, value2):
            self.value1 = value1
            self.value2 = value2

    return Base

Base = make_base(jit)

@jit
class Derived(Base):

    value3 = float_

    @void(int_)
    def setvalue(self, value):
        self.value3 = value

@autojit
def base_typeof():
    obj1 = Base(10, 11.0)
    return numba.typeof(obj1.value1), numba.typeof(obj1.value2)

@autojit
def derived_typeof():
    obj = Derived(10, 11.0)
    return (numba.typeof(obj.value1),
            numba.typeof(obj.value2),
            numba.typeof(obj.value3))

def test_typeof():
    pass
    # TODO: type recognition of extension object instantiation
    # assert base_typeof() == (double, int_), base_typeof()
    # assert derived_typeof() == (double, int_, float_), derived_typeof()


#------------------------------------------------------------------------
# Test Specialized autojit typeof
#------------------------------------------------------------------------

AutoBase = make_base(autojit)

@autojit
def attrtypes(obj):
    return numba.typeof(obj.value1), numba.typeof(obj.value2)

def test_autobase():
    obj = AutoBase(10, 11.0)
    assert attrtypes(obj) == (double, int_)

if __name__ == '__main__':
    test_typeof()
    test_autobase()
